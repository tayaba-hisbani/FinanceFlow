from __future__ import annotations

from typing import Any, Dict

from agents.document_agent import DocumentAgent
from agents.finance_agent import FinanceAgent
from agents.risk_agent import RiskAgent
from agents.reconciliation_agent import ReconciliationAgent
from agents.reporting_agent import ReportingAgent


class Orchestrator:
    """
    Coordinates the FinanceFlow AI multi-agent workflow.

    Workflow:
    1. Document Agent
    2. Finance Agent
    3. Policy/RAG retrieval
    4. Risk Agent
    5. Reconciliation Agent
    6. Reporting Agent
    """

    def __init__(self, gemini, rag):
        self.gemini = gemini
        self.rag = rag

        # Initialize every agent with the dependencies it needs.
        self.document_agent = DocumentAgent(gemini)
        self.finance_agent = FinanceAgent(gemini)
        self.risk_agent = RiskAgent(gemini)
        self.reconciliation_agent = ReconciliationAgent(gemini)

        # IMPORTANT:
        # ReportingAgent requires the Gemini service.
        self.reporting_agent = ReportingAgent(gemini)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        # Make sure trace always exists.
        state.setdefault("trace", [])

        # ---------------------------------------------------------
        # 1. DOCUMENT AGENT
        # ---------------------------------------------------------
        try:
            state = self.document_agent.run(state)

        except Exception as exc:
            state["trace"].append({
                "agent": "Document Agent",
                "status": "FAILED",
                "message": str(exc),
            })

        # ---------------------------------------------------------
        # 2. FINANCE AGENT
        # ---------------------------------------------------------
        try:
            state = self.finance_agent.run(state)

        except Exception as exc:
            state["trace"].append({
                "agent": "Finance Agent",
                "status": "FAILED",
                "message": str(exc),
            })

        # ---------------------------------------------------------
        # 3. POLICY / RAG RETRIEVAL
        # ---------------------------------------------------------
        try:
            queries = self._build_policy_queries(state)

            retrieved_policy = []

            for query in queries:
                results = self.rag.retrieve(query, k=2)

                if results:
                    retrieved_policy.extend(results)

            # Remove duplicate policy chunks.
            unique_policy = []
            seen = set()

            for item in retrieved_policy:
                if isinstance(item, dict):
                    text = item.get("text", "").strip()
                else:
                    text = str(item).strip()

                if text and text not in seen:
                    seen.add(text)
                    unique_policy.append(item)

            state["retrieved_policy"] = unique_policy

            state["trace"].append({
                "agent": "Policy RAG",
                "status": "COMPLETE",
                "message": f"Retrieved {len(unique_policy)} policy evidence items.",
            })

        except Exception as exc:
            state["retrieved_policy"] = []

            state["trace"].append({
                "agent": "Policy RAG",
                "status": "FAILED",
                "message": str(exc),
            })

        # ---------------------------------------------------------
        # 4. RISK AGENT
        # ---------------------------------------------------------
        try:
            state = self.risk_agent.run(state)

        except Exception as exc:
            state["trace"].append({
                "agent": "Risk Agent",
                "status": "FAILED",
                "message": str(exc),
            })

        # ---------------------------------------------------------
        # 5. RECONCILIATION AGENT
        # ---------------------------------------------------------
        try:
            state = self.reconciliation_agent.run(state)

        except Exception as exc:
            state["trace"].append({
                "agent": "Reconciliation Agent",
                "status": "FAILED",
                "message": str(exc),
            })

        # ---------------------------------------------------------
        # 6. REPORTING AGENT
        # ---------------------------------------------------------
        try:
            # IMPORTANT:
            # ReportingAgent.run() returns the updated state.
            state = self.reporting_agent.run(state)

        except Exception as exc:
            state["trace"].append({
                "agent": "Reporting Agent",
                "status": "FAILED",
                "message": str(exc),
            })

            state["reporting_status"] = "FAILED"
            state["report"] = None

        return state

    def _build_policy_queries(self, state: Dict[str, Any]):
        """
        Create policy-search queries based on the current workflow state.
        """

        queries = [
            "invoice approval requirements",
            "duplicate invoice controls",
            "expense evidence requirements",
            "transaction classification",
            "cash flow controls",
        ]

        # Add document-specific information when available.
        documents = state.get("documents", [])

        if documents:
            for document in documents:

                if not isinstance(document, dict):
                    continue

                vendor = document.get("vendor")

                if vendor:
                    queries.append(
                        f"expense and invoice policy for vendor {vendor}"
                    )

                category = document.get("category")

                if category:
                    queries.append(
                        f"expense policy for {category}"
                    )

        return queries
