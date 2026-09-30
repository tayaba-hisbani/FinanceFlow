from dataclasses import dataclass, field
from typing import Any

@dataclass
class DocumentResult:
    filename: str
    text: str
    document_type: str

@dataclass
class AgentResult:
    agent: str
    status: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

@dataclass
class WorkflowState:
    documents: list[DocumentResult] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    rag_context: list[str] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    approval_required: bool = False
    approval_reason: str = ""
    final_report: dict[str, Any] = field(default_factory=dict)
