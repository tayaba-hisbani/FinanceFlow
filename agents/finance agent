from core.models import AgentResult, WorkflowState
from core.finance import normalize_columns, basic_reconciliation

class FinanceAgent:
    name = "Finance Agent"
    def run(self, state: WorkflowState):
        if not state.transactions:
            state.results.append(AgentResult(self.name, "warning", "No transaction table was uploaded."))
            return state
        import pandas as pd
        df = normalize_columns(pd.DataFrame(state.transactions))
        check = basic_reconciliation(df)
        state.results.append(AgentResult(self.name, "complete", "Computed deterministic financial totals and data-quality checks.", check, ["transaction table"]))
        return state

