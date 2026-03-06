import os
import json
import logging
import psycopg2
from typing import TypedDict, NotRequired
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# Default to a lower-cost deployment for workflow status extraction.
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

missing_env = [
    key
    for key, value in {
        "DATABASE_URL": DATABASE_URL,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    }.items()
    if not value
]

if missing_env:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_env)}")

# --- 1. LANGGRAPH STATE & SETUP ---
class AgentState(TypedDict):
    claim_id: int
    image_url: str
    policy_tier: NotRequired[str]
    damage_report: NotRequired[str]
    final_decision: NotRequired[str]

# Initialize Azure LLM
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=220,
)

# --- 2. THE AGENT NODES (REASONING STEPS) ---

def fetch_policy_node(state: AgentState):
    logging.info("Fetching policy for claim_id=%s", state["claim_id"])
    update_claim_progress(state["claim_id"], "fetch_policy")
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.tier FROM policies p
                JOIN claims c ON p.policy_number = c.policy_number
                WHERE c.id = %s
                """,
                (state["claim_id"],),
            )
            row = cur.fetchone()

    if not row:
        raise ValueError(f"No policy found for claim_id={state['claim_id']}")

    tier = row[0]
    return {"policy_tier": tier}

def vision_analysis_node(state: AgentState):
    logging.info("Analyzing image for claim_id=%s", state["claim_id"])
    update_claim_progress(state["claim_id"], "analyzing_image")
    prompt = [
        (
            "system",
            "You are an insurance adjuster. Return concise output under 120 words with: damaged parts, severity (low/medium/high), and one-line risk note.",
        ),
        ("user", [
            {"type": "text", "text": "Analyze this vehicle image for claim triage. Keep it concise."},
            {"type": "image_url", "image_url": {"url": state['image_url']}}
        ])
    ]
    response = llm.invoke(prompt)
    return {"damage_report": response.content}

def auditor_node(state: AgentState):
    logging.info("Auditing claim_id=%s against tier=%s", state["claim_id"], state["policy_tier"])
    update_claim_progress(state["claim_id"], "auditing_claim")
    report = state['damage_report'].lower()
    tier = state['policy_tier']
    
    # Simple logic for the demo
    decision = "approved"
    fraud_score = 0.12
    if "high" in report or "totaled" in report:
        if tier != "Premium":
            decision = "escalated"
            fraud_score = 0.58
        else:
            fraud_score = 0.28

    if "inconsistent" in report or "tamper" in report:
        fraud_score = max(fraud_score, 0.72)
        decision = "escalated"
    
    # damage_assessment is JSON in your DB, so store a JSON payload.
    damage_payload = {
        "summary": state["damage_report"][:1000],
        "agent": "audit_claim",
        "tier": tier,
    }

    # Update Neon DB
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE claims SET status = %s, damage_assessment = %s::json, fraud_score = %s WHERE id = %s",
                (decision, json.dumps(damage_payload), fraud_score, state['claim_id'])
            )
    return {"final_decision": decision}


def update_claim_progress(claim_id: int, status: str):
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE claims SET status = %s WHERE id = %s", (status, claim_id))

# --- 3. CONSTRUCT THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("fetch_policy", fetch_policy_node)
workflow.add_node("analyze_image", vision_analysis_node)
workflow.add_node("audit_claim", auditor_node)

workflow.set_entry_point("fetch_policy")
workflow.add_edge("fetch_policy", "analyze_image")
workflow.add_edge("analyze_image", "audit_claim")
workflow.add_edge("audit_claim", END)

workflow_app = workflow.compile()

# --- 4. FASTAPI WRAPPER ---
app = FastAPI()

class ClaimRequest(BaseModel):
    claimId: int
    imageUrl: str

@app.post("/process-claim")
async def process_claim(request: ClaimRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_agent_logic, request.claimId, request.imageUrl)
    return {"status": "Agent processing started", "claimId": request.claimId}

def run_agent_logic(claim_id, image_url):
    logging.info("Agent workflow started for claim_id=%s", claim_id)
    try:
        workflow_app.invoke({"claim_id": claim_id, "image_url": image_url})
        logging.info("Workflow complete for claim_id=%s", claim_id)
    except Exception:
        update_claim_progress(claim_id, "failed")
        logging.exception("Workflow failed for claim_id=%s", claim_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)