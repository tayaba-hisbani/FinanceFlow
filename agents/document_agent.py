from core.models import AgentResult, WorkflowState
from core.gemini import GeminiService

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "vendor": {"type": "STRING"},
        "invoice_number": {"type": "STRING"},
        "invoice_date": {"type": "STRING"},
        "amount": {"type": "NUMBER"},
        "currency": {"type": "STRING"},
        "tax": {"type": "NUMBER"},
        "description": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["vendor", "invoice_number", "invoice_date", "amount", "currency", "tax", "description", "confidence"],
}

class DocumentAgent:
    name = "Document Agent"
    def run(self, state: WorkflowState, gemini: GeminiService):
        for doc in state.documents:
            prompt = f"""Extract structured invoice/receipt information from the document below.
Do not invent missing values. Use empty strings or 0 when absent. Confidence must be 0-1.
DOCUMENT: {doc.filename}
{doc.text[:30000]}"""
            try:
                data = gemini.generate_json(prompt, SCHEMA)
                state.results.append(AgentResult(self.name, "complete", f"Extracted {doc.filename}.", data, [doc.filename]))
            except Exception as e:
                state.results.append(AgentResult(self.name, "error", f"Extraction failed for {doc.filename}: {e}", {}, [doc.filename]))
        return state

