from io import BytesIO
import pandas as pd
from pypdf import PdfReader
from docx import Document


def extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def extract_docx(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def extract_upload(uploaded_file):
    name = uploaded_file.name
    data = uploaded_file.getvalue()
    lower = name.lower()
    if lower.endswith(".pdf"):
        return extract_pdf(data), "PDF"
    if lower.endswith(".docx"):
        return extract_docx(data), "DOCX"
    raise ValueError(f"Unsupported document type: {name}")


def read_finance_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Finance table must be CSV or XLSX.")
