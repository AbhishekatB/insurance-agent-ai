# Insurance Agent AI

An AI-powered insurance claim processing service built with **LangGraph**, **Azure OpenAI (GPT-4o vision)**, **FastAPI**, and **PostgreSQL (Neon DB)**.

---

## Overview

`main.py` implements a multi-step agentic workflow that automatically triages insurance claims. When a claim is submitted, the agent:

1. Looks up the claimant's policy tier from the database.
2. Runs GPT-4o vision analysis on the submitted vehicle image to produce a damage report.
3. Audits the damage report against the policy tier, calculates a fraud score, and writes a final decision back to the database.

All three steps are wired together as a **LangGraph** state machine and exposed through a single **FastAPI** endpoint.

---

## Architecture

```
POST /process-claim
        │
        ▼
┌───────────────────┐
│  fetch_policy     │  Reads policy tier from PostgreSQL for the given claim ID
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  analyze_image    │  Calls Azure OpenAI GPT-4o with the claim image URL;
│                   │  returns damaged parts, severity, and a risk note
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  audit_claim      │  Compares damage report + policy tier, sets decision
│                   │  (approved / escalated) and fraud score, updates DB
└────────┬──────────┘
         │
         ▼
        END
```

---

## Key Components in `main.py`

### 1. Environment & Configuration

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (Neon DB or any Postgres instance) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key (also accepted as `AZURE_OPENAI_KEY`) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (defaults to `gpt-4o-mini`) |

The app raises a `RuntimeError` at startup if any required variable is missing.

### 2. LangGraph State — `AgentState`

```python
class AgentState(TypedDict):
    claim_id: int          # The claim being processed
    image_url: str         # URL of the vehicle image
    policy_tier: str       # Populated by fetch_policy node
    damage_report: str     # Populated by analyze_image node
    final_decision: str    # Populated by audit_claim node
```

This typed dictionary is passed between nodes as the shared state of the workflow.

### 3. Agent Nodes

#### `fetch_policy_node`
- Updates the claim status to `fetch_policy` in the database.
- Queries `policies` joined with `claims` to retrieve the policy tier (`Basic`, `Standard`, `Premium`).
- Raises `ValueError` if no matching policy is found.

#### `vision_analysis_node`
- Updates the claim status to `analyzing_image`.
- Sends the image URL to Azure OpenAI using a multimodal prompt.
- The LLM returns a concise report: damaged parts, severity (`low` / `medium` / `high`), and a one-line risk note.

#### `auditor_node`
- Updates the claim status to `auditing_claim`.
- Applies rule-based logic on top of the LLM report:
  - High severity or total loss on a non-Premium policy → `escalated`, higher fraud score.
  - Keywords like `inconsistent` or `tamper` in the report → fraud score ≥ 0.72, `escalated`.
  - Otherwise → `approved`, low fraud score.
- Writes the final `status`, `damage_assessment` (JSON), and `fraud_score` to the `claims` table.

### 4. LangGraph Workflow

```python
workflow = StateGraph(AgentState)
workflow.add_node("fetch_policy",  fetch_policy_node)
workflow.add_node("analyze_image", vision_analysis_node)
workflow.add_node("audit_claim",   auditor_node)

workflow.set_entry_point("fetch_policy")
workflow.add_edge("fetch_policy",  "analyze_image")
workflow.add_edge("analyze_image", "audit_claim")
workflow.add_edge("audit_claim",   END)

workflow_app = workflow.compile()
```

Nodes run sequentially; the compiled `workflow_app` is invoked once per claim.

### 5. FastAPI Endpoint

#### `POST /process-claim`

**Request body:**
```json
{
  "claimId": 42,
  "imageUrl": "https://example.com/car-damage.jpg"
}
```

**Response:**
```json
{
  "status": "Agent processing started",
  "claimId": 42
}
```

Processing is kicked off as a **background task** (`BackgroundTasks`), so the HTTP response returns immediately while the LangGraph workflow runs asynchronously. If the workflow fails, the claim status is set to `failed` and the exception is logged.

---

## Database Schema (expected)

```sql
-- policies table
CREATE TABLE policies (
    policy_number TEXT PRIMARY KEY,
    tier          TEXT  -- e.g. 'Basic', 'Standard', 'Premium'
);

-- claims table
CREATE TABLE claims (
    id                SERIAL PRIMARY KEY,
    policy_number     TEXT REFERENCES policies(policy_number),
    status            TEXT,
    damage_assessment JSON,
    fraud_score       FLOAT
);
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A running PostgreSQL instance (or [Neon](https://neon.tech) serverless Postgres)
- An Azure OpenAI resource with a GPT-4o / GPT-4o-mini deployment

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@host/dbname
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini   # optional, defaults to gpt-4o-mini
```

### Running the Server

```bash
python main.py
```

The API will be available at `http://0.0.0.0:8080`.

You can also run it with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Example Request

```bash
curl -X POST http://localhost:8080/process-claim \
     -H "Content-Type: application/json" \
     -d '{"claimId": 1, "imageUrl": "https://example.com/damage.jpg"}'
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` / `uvicorn` | HTTP server and ASGI framework |
| `langgraph` | Stateful multi-agent workflow orchestration |
| `langchain-openai` | Azure OpenAI LLM integration |
| `langchain-core` | Core LangChain primitives |
| `psycopg2-binary` | PostgreSQL database driver |
| `python-dotenv` | `.env` file loading |
| `azure-storage-blob` | Azure Blob Storage (for image hosting if needed) |
| `azure-identity` | Azure credential management |
| `pydantic` | Request/response validation |
