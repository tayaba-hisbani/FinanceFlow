from core.models import AgentResult, WorkflowState

class ReconciliationAgent:
    name = "Reconciliation Agent"
    def run(self, state: WorkflowState):
        extracted = [r.data for r in state.results if r.agent == "Document Agent" and r.status == "complete"]
        finance = next((r.data for r in state.results if r.agent == "Finance Agent"), {})
        total = finance.get("summary", {}).get("total", 0)
        invoice_total = sum(float(x.get("amount", 0) or 0) for x in extracted)
        difference = round(total - invoice_total, 2)
        data = {"transaction_total": total, "extracted_invoice_total": invoice_total, "difference": difference}
        status = "match" if abs(difference) < 0.01 else "review"
        if status == "review":
            state.approval_required = True
            state.approval_reason += f" Reconciliation difference: {difference}."
        state.results.append(AgentResult(self.name, status, f"Reconciliation status: {status}.", data, ["transaction table", "extracted documents"]))
        return state

