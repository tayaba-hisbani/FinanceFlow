import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.extractors import extract_upload, read_finance_table
from core.models import WorkflowState, DocumentResult
from core.rag import PolicyRAG
from core.gemini import GeminiService
from agents.orchestrator import Orchestrator

st.set_page_config(page_title="FinanceFlow AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.5rem;}
.hero {padding: 28px 32px; border:1px solid #20334A; border-radius:24px; background:linear-gradient(135deg,#0E2437,#08111D); margin-bottom:20px;}
.hero h1 {font-size:46px; margin:0; letter-spacing:-1.5px;}
.hero p {color:#A8B8CC; font-size:17px; margin:8px 0 0;}
.badge {display:inline-block; padding:5px 10px; border-radius:999px; background:#12372F; color:#6EE7B7; font-size:12px; font-weight:700; margin-bottom:10px;}
.card {padding:18px; border:1px solid #20334A; border-radius:18px; background:#0D1B2A;}
.small {color:#8FA2B8; font-size:13px;}
.stButton>button {border-radius:12px; font-weight:700; min-height:44px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
<div class='badge'>MULTI-AGENT FINANCE OPERATIONS</div>
<h1>FinanceFlow AI</h1>
<p>From messy financial documents to verified insights, exception handling and management reporting.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ Control Center")
    st.caption("Configure the AI workflow and provide evidence. Never put your API key in GitHub code.")
    api_key = st.text_input("Gemini API key", type="password", value=st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")))
    model = st.text_input("Gemini model", value=st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash")))
    st.divider()
    st.markdown("**Workflow**")
    for x in ["Document extraction", "Policy RAG", "Finance checks", "Risk detection", "Reconciliation", "Management report", "Human approval"]:
        st.write("✓", x)

if "state" not in st.session_state:
    st.session_state.state = None

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='card'><b>1 · Evidence</b><br><span class='small'>Invoices, receipts and finance tables</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><b>2 · Agentic workflow</b><br><span class='small'>Specialized agents verify and reconcile</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><b>3 · Decision</b><br><span class='small'>Report + exceptions + approval gate</span></div>", unsafe_allow_html=True)
st.write("")

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Upload financial evidence")
    docs = st.file_uploader("PDF / DOCX invoices or receipts", type=["pdf", "docx"], accept_multiple_files=True)
    table_file = st.file_uploader("CSV / XLSX transaction register", type=["csv", "xlsx"])
with right:
    st.subheader("Workflow policy")
    st.info("The included company policy is the RAG knowledge base. Replace data/company_policy.txt with your demo company's rules.")
    run = st.button("▶ Run FinanceFlow", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("Add a Gemini API key in the sidebar or Streamlit Secrets first.")
        st.stop()
    if not docs and not table_file:
        st.warning("Upload at least one document or a transaction table.")
        st.stop()
    try:
        state = WorkflowState()
        for f in docs or []:
            text, kind = extract_upload(f)
            state.documents.append(DocumentResult(f.name, text, kind))
        if table_file:
            df = read_finance_table(table_file)
            state.transactions = df.to_dict(orient="records")
        rag = PolicyRAG("data/company_policy.txt")
        gemini = GeminiService(api_key=api_key, model=model)
        with st.status("Running multi-agent workflow...", expanded=True) as status:
            st.write("Retrieving policy evidence…")
            st.write("Document Agent → Finance Agent → Risk Agent → Reconciliation Agent → Reporting Agent")
            state = Orchestrator(gemini, rag).run(state)
            status.update(label="Workflow complete", state="complete")
        st.session_state.state = state
        st.rerun()
    except Exception as e:
        st.exception(e)

state = st.session_state.state
if state:
    st.divider()
    st.subheader("Mission Control")
    cols = st.columns(4)
    finance_result = next((r for r in state.results if r.agent == "Finance Agent"), None)
    recon_result = next((r for r in state.results if r.agent == "Reconciliation Agent"), None)
    total = finance_result.data.get("summary", {}).get("total", 0) if finance_result else 0
    diff = recon_result.data.get("difference", 0) if recon_result else 0
    cols[0].metric("Transaction total", f"PKR {total:,.2f}")
    cols[1].metric("Documents", len(state.documents))
    cols[2].metric("Reconciliation", "MATCH" if abs(diff) < 0.01 else f"PKR {diff:,.2f} gap")
    cols[3].metric("Approval", "REQUIRED" if state.approval_required else "CLEAR")

    tabs = st.tabs(["Executive Report", "Agent Trace", "Evidence & RAG", "Approval Gate"])
    with tabs[0]:
        report = state.final_report
        if report:
            st.markdown(f"### Executive summary\n{report.get('executive_summary','')}")
            a, b = st.columns(2)
            with a:
                st.markdown("#### Key findings")
                for item in report.get("key_findings", []): st.write("•", item)
            with b:
                st.markdown("#### Recommended actions")
                for item in report.get("recommended_actions", []): st.write("•", item)
            st.markdown("#### Approval note")
            st.warning(report.get("approval_note", "")) if state.approval_required else st.success(report.get("approval_note", "No approval exception detected."))
        else:
            st.warning("No report was generated. Inspect Agent Trace for the failing stage.")

        if finance_result and finance_result.data.get("summary"):
            st.markdown("#### Transaction overview")
            s = finance_result.data["summary"]
            fig = go.Figure(go.Indicator(mode="gauge+number", value=s.get("total",0), title={"text":"Transaction value"}, gauge={"axis":{"visible":True}}))
            fig.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        for r in state.results:
            icon = "🟢" if r.status in ["complete", "match"] else "🟠" if r.status in ["warning", "review"] else "🔴"
            with st.expander(f"{icon} {r.agent} · {r.status}"):
                st.write(r.summary)
                if r.data: st.json(r.data)
                if r.evidence: st.caption("Evidence: " + ", ".join(r.evidence))

    with tabs[2]:
        st.markdown("#### Retrieved policy evidence")
        for i, chunk in enumerate(state.rag_context, 1):
            st.markdown(f"**Policy chunk {i}**")
            st.info(chunk)
        st.markdown("#### Uploaded document text")
        for d in state.documents:
            with st.expander(d.filename): st.text(d.text[:10000])

    with tabs[3]:
        if state.approval_required:
            st.warning("Human approval required before treating the workflow as cleared.")
            st.markdown(f"**Reason:** {state.approval_reason}")
            approve = st.checkbox("I reviewed the flagged exceptions and approve this demo result.")
            if approve:
                st.success("Human approval recorded for this session.")
        else:
            st.success("No approval exception was triggered by the deterministic checks.")

    st.download_button("Download workflow audit JSON", json.dumps({"approval_required": state.approval_required, "approval_reason": state.approval_reason, "results": [{"agent":r.agent,"status":r.status,"summary":r.summary,"data":r.data,"evidence":r.evidence} for r in state.results], "report": state.final_report}, indent=2), file_name="financeflow_audit.json", mime="application/json")
