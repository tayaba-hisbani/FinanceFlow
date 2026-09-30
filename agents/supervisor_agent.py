from core.models import AgentResult, WorkflowState

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reason": {"type": "STRING"},
        "steps": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["reason", "steps"]
}

ALLOWED = {"document_extraction", "finance_analysis", "risk_analysis", "reconciliation", "reporting"}

class SupervisorAgent:
    """Planner agent: Gemini proposes a workflow, but code validates every action against an allow-list."""
    name = "Supervisor Agent"
    def run(self, state: WorkflowState, gemini):
        available = {
            "documents": [d.filename for d in state.documents],
            "transaction_table": bool(state.transactions),
        }
        prompt = f"""You are the workflow supervisor for a finance operations system.
Choose the minimum safe sequence from these allowed steps only:
{sorted(ALLOWED)}
Rules: document_extraction is useful when documents exist; finance_analysis when a transaction table exists; risk_analysis after available evidence; reconciliation requires both extracted documents and transactions; reporting is last.
Never invent a step. Return a short reason and an ordered list.
AVAILABLE INPUTS: {available}"""
        try:
            plan = gemini.generate_json(prompt, SCHEMA)
            steps = [s for s in plan.get("steps", []) if s in ALLOWED]
        except Exception:
            plan = {"reason": "Fallback safety plan", "steps": []}
            steps = []
        if not steps:
            steps = ["document_extraction"] if state.documents else []
            if state.transactions: steps.append("finance_analysis")
            if state.documents or state.transactions: steps.append("risk_analysis")
            if state.documents and state.transactions: steps.append("reconciliation")
            steps.append("reporting")
        # Enforce dependencies regardless of model output.
        ordered = []
        for step in ["document_extraction", "finance_analysis", "risk_analysis", "reconciliation", "reporting"]:
            if step in steps and step not in ordered:
                ordered.append(step)
        state.results.append(AgentResult(self.name, "complete", plan.get("reason", "Workflow plan created."), {"plan": ordered}, ["available inputs"]))
        return state, ordered
