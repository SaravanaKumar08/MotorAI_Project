# MotorAI - Intelligent Motor Insurance Platform

AI/ML platform for motor insurance built on the freMTPL2 French motor dataset.

## What it does

| Feature | Endpoint | Model |
|---|---|---|
| Claim frequency (claims per year) | `POST /predict/frequency` | Random Forest |
| Typical claim amount | `POST /predict/severity` | Random Forest (MAE) |
| Customer segment | `POST /predict/segment` | MiniBatch K-Means (3 clusters) |
| Combined claim triage (used by n8n) | `POST /predict/triage` | all three |
| Policy assistant (RAG over the policy PDF) | `POST /chat` | TF-IDF retrieval + optional Ollama llama3.2 |

Honest note: the claim-amount model does not beat a median baseline
(see `reports/individual_claim_model_mae_comparison.csv`); use it as a typical-size signal only.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements-dev.txt
python -m pytest tests/test_api.py -q
uvicorn api.main:app --reload       # http://127.0.0.1:8000/docs
```

Optional LLM answers for `/chat` (Ollama must be running):

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

## Deploy

- GitHub: push this repo (only the three small models in `artifacts/` are committed).
- Render: New > Blueprint > pick the repo (`render.yaml`).
- n8n: import `n8n/motorai_claim_workflow.json` and set your Render URL.

Full walkthrough: `docs/MotorAI_Step_by_Step_Guide.docx`.

## Project layout

- `api/` FastAPI app (routes, services, schemas)
- `src/` data preparation, training, NLP and evaluation scripts
- `notebooks/` analysis notebooks 01-07
- `artifacts/` trained models and FAISS index
- `reports/` evaluation CSVs and charts
- `n8n/` automation workflow
