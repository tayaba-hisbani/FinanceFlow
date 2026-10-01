from __future__ import annotations

from typing import Any, Dict, List


class ReportingAgent:
    """
    Reporting Agent.

    Primary path:
        Gemini generates an executive report.

    Fallback path:
        A deterministic evidence-based report is generated
        if Gemini fails.

    This prevents the entire FinanceFlow workflow from
    failing because of an LLM/API problem.
    """

    def __init__(self, gemini):
        self.gemini = gemini

    def run(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:

        finance = state.get(
            "finance",
            {}
        )

        risk = state.get(
            "risk",
            {}
        )

        reconciliation = state.get(
            "reconciliation",
            {}
        )

        documents = state.get(
            "documents",
            []
        )

        retrieved_policy = state.get(
            "retrieved_policy",
            []
        )

        evidence = {
            "finance": finance,
            "risk": risk,
            "reconciliation": reconciliation,
            "documents": documents,
            "policy": retrieved_policy,
        }

        # --------------------------------------------------
        # Try Gemini first
        # --------------------------------------------------

        try:

            prompt = f"""
You are the Reporting Agent for FinanceFlow AI.

Create a concise professional management report.

IMPORTANT RULES:

1. Use ONLY the evidence provided below.
2. Do not invent financial numbers.
3. Do not invent transactions.
4. Do not invent vendors.
5. Do not claim fraud unless the evidence explicitly
   supports a fraud concern.
6. Clearly distinguish facts from recommendations.
7. Mention items requiring human approval.
8. Include reconciliation results.
9. Include important risk flags.
10. Keep the report suitable for a small-business manager.

Evidence:

{evidence}
"""

            schema = {
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string"
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "risk_summary": {
                        "type": "string"
                    },
                    "reconciliation_summary": {
                        "type": "string"
                    },
                    "recommended_actions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "approval_note": {
                        "type": "string"
                    }
                },
                "required": [
                    "executive_summary",
                    "key_findings",
                    "risk_summary",
                    "reconciliation_summary",
                    "recommended_actions",
                    "approval_note"
                ]
            }

            report = self.gemini.generate_json(
                prompt=prompt,
                schema=schema,
                system_instruction=(
                    "You are a finance operations reporting "
                    "agent. Be factual, concise and evidence-based."
                )
            )

            # Make sure Gemini actually returned useful data.
            if not report.get("executive_summary"):
                raise RuntimeError(
                    "Gemini report was empty."
                )

            state["report"] = report
            state["report_source"] = "Gemini"
            state["reporting_status"] = "COMPLETE"

            state.setdefault(
                "trace",
                []
            ).append(
                {
                    "agent": "Reporting Agent",
                    "status": "COMPLETE",
                    "message": (
                        "Executive report generated "
                        "using Gemini."
                    )
                }
            )

            return state

        except Exception as exc:

            # --------------------------------------------------
            # Gemini failed → deterministic fallback
            # --------------------------------------------------

            fallback = self._fallback_report(
                finance=finance,
                risk=risk,
                reconciliation=reconciliation,
                documents=documents
            )

            state["report"] = fallback
            state["report_source"] = "Deterministic fallback"
            state["reporting_status"] = "FALLBACK"

            state.setdefault(
                "trace",
                []
            ).append(
                {
                    "agent": "Reporting Agent",
                    "status": "FALLBACK",
                    "message": (
                        "Gemini report generation failed. "
                        "Evidence-based fallback report created."
                    ),
                    "error": str(exc)
                }
            )

            return state

    # ======================================================
    # FALLBACK REPORT
    # ======================================================

    def _fallback_report(
        self,
        finance: Dict[str, Any],
        risk: Dict[str, Any],
        reconciliation: Dict[str, Any],
        documents: List[Any]
    ) -> Dict[str, Any]:

        # -------------------------------
        # Financial information
        # -------------------------------

        total = finance.get(
            "total_amount",
            finance.get(
                "total",
                0
            )
        )

        transaction_count = finance.get(
            "transaction_count",
            finance.get(
                "count",
                0
            )
        )

        currency = finance.get(
            "currency",
            "PKR"
        )

        # -------------------------------
        # Risk information
        # -------------------------------

        risk_flags = (
            risk.get("flags")
            or risk.get("risk_flags")
            or []
        )

        risk_level = risk.get(
            "risk_level",
            "Review"
        )

        # -------------------------------
        # Reconciliation
        # -------------------------------

        recon_status = reconciliation.get(
            "status",
            reconciliation.get(
                "result",
                "Not available"
            )
        )

        recon_difference = reconciliation.get(
            "difference",
            reconciliation.get(
                "amount_difference",
                0
            )
        )

        # -------------------------------
        # Documents
        # -------------------------------

        document_count = len(
            documents
        )

        # -------------------------------
        # Findings
        # -------------------------------

        findings = []

        findings.append(
            f"Processed {document_count} uploaded "
            f"financial document(s)."
        )

        findings.append(
            f"Analyzed {transaction_count} transaction(s)."
        )

        findings.append(
            f"Calculated transaction value: "
            f"{currency} {total:,.2f}."
        )

        findings.append(
            f"Reconciliation status: {recon_status}."
        )

        if risk_flags:

            for flag in risk_flags[:5]:

                if isinstance(flag, dict):

                    description = (
                        flag.get("message")
                        or flag.get("description")
                        or str(flag)
                    )

                else:
                    description = str(flag)

                findings.append(
                    f"Risk flag: {description}"
                )

        else:

            findings.append(
                "No explicit risk flags were returned "
                "by the Risk Agent."
            )

        # -------------------------------
        # Recommended actions
        # -------------------------------

        recommendations = []

        if risk_flags:

            recommendations.append(
                "Review all flagged transactions "
                "before final approval."
            )

        if str(recon_status).lower() not in {
            "matched",
            "match",
            "reconciled"
        }:

            recommendations.append(
                "Investigate reconciliation differences "
                "before closing the accounting period."
            )

        if not recommendations:

            recommendations.append(
                "Continue routine financial review "
                "and retain supporting documents."
            )

        # -------------------------------
        # Approval
        # -------------------------------

        if risk_flags or (
            str(recon_status).lower()
            not in {
                "matched",
                "match",
                "reconciled"
            }
        ):

            approval_note = (
                "Human review is recommended before "
                "finalizing the flagged financial records."
            )

        else:

            approval_note = (
                "No exception requiring mandatory "
                "human approval was identified by "
                "the available evidence."
            )

        # -------------------------------
        # Executive summary
        # -------------------------------

        if risk_flags:

            summary = (
                f"FinanceFlow processed {document_count} "
                f"document(s) and {transaction_count} "
                f"transaction(s), with a total analyzed "
                f"value of {currency} {total:,.2f}. "
                f"The Risk Agent identified "
                f"{len(risk_flags)} risk flag(s), so "
                f"human review is recommended."
            )

        else:

            summary = (
                f"FinanceFlow processed {document_count} "
                f"document(s) and {transaction_count} "
                f"transaction(s), with a total analyzed "
                f"value of {currency} {total:,.2f}. "
                f"No explicit risk flags were returned "
                f"by the Risk Agent."
            )

        # -------------------------------
        # Reconciliation summary
        # -------------------------------

        reconciliation_summary = (
            f"Reconciliation result: {recon_status}. "
            f"Recorded difference: "
            f"{currency} {recon_difference:,.2f}."
        )

        # -------------------------------
        # Final report
        # -------------------------------

        return {
            "executive_summary": summary,

            "key_findings": findings,

            "risk_summary": (
                f"Risk level: {risk_level}. "
                f"Risk flags identified: "
                f"{len(risk_flags)}."
            ),

            "reconciliation_summary": (
                reconciliation_summary
            ),

            "recommended_actions": (
                recommendations
            ),

            "approval_note": approval_note
        }
