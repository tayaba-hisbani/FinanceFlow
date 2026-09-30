from core.models import AgentResult, WorkflowState

class RiskAgent:
    name = "Risk Agent"
    def run(self, state: WorkflowState):
        risks = []
        # Deterministic checks are used for financial controls; Gemini is not asked to invent thresholds.
        for r in state.results:
            if r.agent == "Finance Agent":
                if r.data.get("duplicate_count", 0) > 0:
                    risks.append("Duplicate transaction/invoice identifiers require review.")
                risks.extend(r.data.get("issues", []))
            if r.agent == "Document Agent":
                conf = float(r.data.get("confidence", 1) or 0)
                amount = float(r.data.get("amount", 0) or 0)
                if conf < 0.75:
                    risks.append(f"Low extraction confidence for {r.evidence[0] if r.evidence else 'document'}.")
                if amount > 100000:
                    risks.append(f"Invoice amount PKR {amount:,.2f} exceeds the policy approval threshold of PKR 100,000.")
                if not r.data.get("invoice_number"):
                    risks.append(f"Invoice number is missing for {r.evidence[0] if r.evidence else 'document'}.")

        # A document is evidence for reimbursement; if a transaction table exists with no source documents,
        # flag the gap rather than pretending the transaction is verified.
        if state.transactions and not state.documents:
            risks.append("Transactions were uploaded without invoice/receipt evidence; finance review is required.")

        state.approval_required = bool(risks)
        state.approval_reason = " ".join(risks) if risks else "No deterministic high-risk exception was detected."
        state.results.append(AgentResult(self.name, "complete", state.approval_reason, {"risks": risks}, risks))
        return state
