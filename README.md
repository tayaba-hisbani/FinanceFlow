# FinanceFlow AI

A hackathon-oriented multi-agent finance operations workflow built with Streamlit and Gemini.

## Architecture

Upload evidence → local Policy RAG → Document Agent → Finance Agent → Risk Agent → Reconciliation Agent → Reporting Agent → Human approval.

The system intentionally keeps calculations and duplicate checks deterministic in Python while using Gemini for document extraction and management-language generation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-3.8-flash"
```

## Demo data

Use a PDF/DOCX invoice and a CSV/XLSX transaction register. The policy RAG source is `data/company_policy.txt`.

## Deployment

Push this repository to GitHub. In Streamlit Community Cloud choose the repository, branch and `app.py`, then put the same secrets into Advanced settings → Secrets.
