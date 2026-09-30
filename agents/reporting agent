from core.models import AgentResult, WorkflowState
from core.gemini import GeminiService

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "executive_summary": {"type": "STRING"},
        "key_findings": {"type": "ARRAY", "items": {"type": "STRING"}},
        "recommended_actions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "approval_note": {"type": "STRING"},
    },
    "required": ["executive_summary", "key_findings", "recommended_actions", "approval_note"],
}

class ReportingAgent:
    name = "Reporting Agent"
    def run(self, state: WorkflowState, gemini: GeminiService):
        context = [
            {"agent": r.agent, "status": r.status, "summary": r.summary, "data": r.data}
            for r in state.results
        ]
        policy = "\n\n".join(state.rag_context) if state.rag_context else "No policy evidence retrieved."
        prompt = f"""You are a finance operations reporting agent. Generate a concise management report.
Use ONLY the evidence supplied below. Do not invent financial facts. Clearly label recommendations as recommendations.
Approval required: {state.approval_required}
Approval reason: {state.approval_reason}
Policy evidence:\n{policy}\n
Agent evidence:\n{context}"""
        try:
            report = gemini.generate_json(prompt, SCHEMA)
            state.final_report = report
            state.results.append(AgentResult(self.name, "complete", "Generated management report.", report, ["agent evidence", "policy evidence"]))
        except Exception as e:
            state.results.append(AgentResult(self.name, "error", f"Report generation failed: {e}"))
        return state

