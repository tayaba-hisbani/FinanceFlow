from core.models import WorkflowState
from agents.supervisor_agent import SupervisorAgent
from agents.document_agent import DocumentAgent
from agents.finance_agent import FinanceAgent
from agents.risk_agent import RiskAgent
from agents.reconciliation_agent import ReconciliationAgent
from agents.reporting_agent import ReportingAgent

class Orchestrator:
    """Deterministic workflow controller. Agents are specialized; the controller controls order and state."""
    def __init__(self, gemini, rag):
        self.gemini = gemini
        self.rag = rag
        self.supervisor_agent = SupervisorAgent()
        self.document_agent = DocumentAgent()
        self.finance_agent = FinanceAgent()
        self.risk_agent = RiskAgent()
        self.reconciliation_agent = ReconciliationAgent()
        self.reporting_agent = ReportingAgent()

    def run(self, state: WorkflowState):
        # 1. retrieve policy evidence first: this is the RAG stage
        queries = ["invoice approval spending limits duplicate invoices documentation expenses", "expense categorization reimbursement policy"]
        retrieved = []
        for q in queries:
            retrieved.extend(self.rag.retrieve(q, k=2))
        seen = set()
        state.rag_context = [x["text"] for x in retrieved if not (x["text"] in seen or seen.add(x["text"]))]

        # 2. Supervisor Agent creates the plan; dependency order is enforced by the controller.
        state, plan = self.supervisor_agent.run(state, self.gemini)
        for step in plan:
            if step == "document_extraction":
                state = self.document_agent.run(state, self.gemini)
            elif step == "finance_analysis":
                state = self.finance_agent.run(state)
            elif step == "risk_analysis":
                state = self.risk_agent.run(state)
            elif step == "reconciliation":
                state = self.reconciliation_agent.run(state)
            elif step == "reporting":
                state = self.reporting_agent.run(state, self.gemini)
        return state

