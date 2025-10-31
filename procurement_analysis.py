# app.py
import os
import io
import re
import ast
import time
import numpy as np
import pandas as pd
import streamlit as st

from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from groq import Groq

# ============================================================================
# Streamlit Page Setup
# ============================================================================
st.set_page_config(
    page_title="TheAiExtract-Textractor",
    page_icon="assets/bot.PNG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Secrets & Env Helpers
# ============================================================================

def get_secret(name: str, default: str | None = None) -> str | None:
    """Prefer OS env, then st.secrets, then default."""
    val = os.getenv(name)
    if val is not None:
        return val
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

LOGIN_USER = get_secret("APP_USERNAME")
LOGIN_PASS = get_secret("APP_PASSWORD")
PREFILL_GROQ = get_secret("GROQ_API_KEY")  

# ============================================================================
# App State
# ============================================================================
if "authed" not in st.session_state:
    st.session_state.authed = False
if "groq_valid" not in st.session_state:
    st.session_state.groq_valid = False
if "groq_client" not in st.session_state:
    st.session_state.groq_client = None
if "ocr" not in st.session_state:
    # Initialize once (faster subsequent OCR calls)
    st.session_state.ocr = PaddleOCR(use_angle_cls=True, lang="en")

def record_groq_error(err: Exception, context: str = "", payload: dict | None = None):
    st.session_state.groq_last_error = {"context": context, "error": str(err)}
    if payload:
        st.session_state.groq_last_request = payload
# ============================================================================
# Domain Data
# ============================================================================
SALESFORCE_COLUMNS = {
    "Opportunity": ["Name", "AccountId", "CloseDate", "StageName", "Amount", "Probability", "OwnerId", "Description"],
    "OpportunityLineItem": ["OpportunityId", "PricebookEntryId", "Quantity", "UnitPrice", "TotalPrice", "Discount", "Description"],
    "Quote": ["QuoteNumber", "OpportunityId", "ValidUntil", "Status", "TotalPrice", "Description"],
    "QuoteLineItem": ["QuoteId", "ProductCode", "Quantity", "UnitPrice", "TotalPrice", "Discount", "Description"],
    "Order": ["OrderNumber", "AccountId", "EffectiveDate", "Status", "TotalAmount", "Description"],
    "OrderItem": ["OrderId", "ProductCode", "Quantity", "UnitPrice", "TotalPrice", "Discount", "Description"],
}

# ============================================================================
# Auth & Validation
# ============================================================================
def validate_login(username: str, password: str) -> bool:
    if not LOGIN_USER or not LOGIN_PASS:
        st.error("Server is missing APP_USERNAME / APP_PASSWORD. Set them via env vars or Streamlit secrets.")
        return False
    return username == LOGIN_USER and password == LOGIN_PASS

def make_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

def validate_groq_key(api_key: str) -> bool:
    try:
        client = make_groq_client(api_key)
        # lightweight ping
        _ = client.models.list()
        st.session_state.groq_client = client
        st.session_state.groq_last_error = None
        return True
    except Exception as e:
        st.session_state.groq_client = None
        record_groq_error(e, context="models.list() during validation")
        st.exception(e)  # shows full traceback in the app
        st.error("Groq API key validation failed. See traceback above.")
        return False

# ============================================================================
# OCR & Extraction
# ============================================================================
def pdf_to_images(pdf_path: str):
    """Return list of PIL images for preview (use first 2 pages)."""
    return convert_from_path(pdf_path)

def pdf_to_text(pdf_path: str) -> str:
    """OCR all pages with PaddleOCR and concatenate text."""
    images = convert_from_path(pdf_path, fmt="jpeg")
    all_text = []
    for img in images:
        img_np = np.array(img)
        result = st.session_state.ocr.ocr(img_np, cls=True)
        for line in result:
            all_text.append(" ".join([w[1][0] for w in line]))
    return "\n".join(all_text)

def classify_doc(text: str, groq_client: Groq) -> str:
    lowered = text.lower()
    if "purchase order" in lowered or "order number" in lowered or "po number" in lowered:
        return "PO"
    if "request for quotation" in lowered or "rfq" in lowered:
        return "RFQ"
    if "quotation" in lowered or "quote" in lowered:
        return "Quotation"

    prompt = (
        "Classify this document as 'RFQ', 'Quotation', or 'Purchase Order (PO)'. "
        "Only respond with one word (RFQ, Quotation, PO). "
        f"Document Content:\n{text[:2000]}"
    )
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    doc_type = resp.choices[0].message.content.strip()
    if doc_type.lower().startswith("purchase"):
        doc_type = "PO"
    return doc_type

def extract_fields(text: str, doc_type: str, salesforce_columns: dict, groq_client: Groq) -> dict:
    if doc_type == "RFQ":
        fields = salesforce_columns["Opportunity"] + ["OpportunityLineItem"]
    elif doc_type == "Quotation":
        fields = salesforce_columns["Quote"] + ["QuoteLineItem"]
    elif doc_type == "PO":
        fields = salesforce_columns["Order"] + ["OrderItems"]
    else:
        fields = []

    prompt = f"Extract the following fields: {fields}\n\nOCR Text:\n{text}"
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
    )
    output = resp.choices[0].message.content.strip()
    output = re.sub(r"```[\s\S]*?```", "", output)
    match = re.search(r"\{[\s\S]+\}", output)
    dict_str = match.group(0) if match else output

    try:
        data = ast.literal_eval(dict_str)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return data

def clean_salesforce_output(extracted: dict, doc_type: str, salesforce_columns: dict) -> dict:
    if doc_type == "PO":
        main_fields = salesforce_columns["Order"]
        line_item_field = "OrderItems"
    elif doc_type == "RFQ":
        main_fields = salesforce_columns["Opportunity"]
        line_item_field = "OpportunityLineItem"
    elif doc_type == "Quotation":
        main_fields = salesforce_columns["Quote"]
        line_item_field = "QuoteLineItem"
    else:
        main_fields = list(extracted.keys())
        line_item_field = None

    clean = {}
    for k, v in extracted.items():
        if k in main_fields or (line_item_field and k == line_item_field):
            clean[k] = v
    return clean

# ============================================================================
# UI Pages
# ============================================================================
def sidebar_header():
    with st.sidebar:
        try:
            st.image("assets/logo.png", width="stretch")
        except Exception:
            st.write("")  # no-op if missing
        st.title("TheAiExtract-Textractor")
        st.caption("Salesforce PDF Extractor • Upload up to 3 PDFs.")

        if st.session_state.authed and st.session_state.groq_valid:
            if st.button("Log out", type="secondary"):
                st.session_state.clear()
                st.rerun()

def page_login():
    sidebar_header()
    st.title("Sign in")

    with st.form("login_form"):
        u = st.text_input("Username", placeholder="Enter username")
        p = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if validate_login(u, p):
            st.session_state.authed = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid credentials.")

def page_groq_key():
    sidebar_header()
    st.title("Connect Groq")

    st.caption("Enter a valid Groq API key to continue.")
    key = st.text_input(
        "Groq API Key",
        type="password",
        value=PREFILL_GROQ or "",
        placeholder="gsk_********************************",
        help="Used only for this session to run classification & field extraction.",
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Validate Key", width="stretch"):
            if key.strip():
                if validate_groq_key(key.strip()):
                    st.session_state.groq_valid = True
                    st.success("Groq key validated. You're good to go!")
                    st.rerun()
            else:
                st.warning("Please paste your Groq API key.")
    with colB:
        if st.button("Test Groq (list models)", width="stretch") and st.session_state.groq_client:
            try:
                models = st.session_state.groq_client.models.list()
                st.success("Groq reachable.")
                st.json(models.dict() if hasattr(models, "dict") else models)
            except Exception as e:
                record_groq_error(e, context="manual models.list() test")
                st.exception(e) 

def page_main():
    sidebar_header()
    st.header("PDF Upload & Extraction")

    uploaded_files = st.file_uploader(
        "Upload up to 3 PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload up to 3 PDF files.")
        return

    if len(uploaded_files) > 3:
        st.warning("Please upload a maximum of 3 files!")
        return

    tabs = st.tabs([f"{file.name}" for file in uploaded_files])

    for idx, uploaded_file in enumerate(uploaded_files):
        with tabs[idx]:
            col1, col2 = st.columns([1.1, 1.5], gap="medium")

            # --- PDF Preview ---
            with col1:
                st.subheader("PDF Preview")
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Rendering PDF..."):
                    try:
                        images = pdf_to_images(temp_path)
                        for i, img in enumerate(images[:2]):
                            st.image(img, caption=f"Page {i+1}")
                    except Exception as e:
                        st.error(f"Could not render PDF. Error: {e}")

            # --- Extraction / Load Results ---
            with col2:
                st.subheader("Extraction Results")
                base_pdf_name = os.path.splitext(uploaded_file.name)[0]
                excel_path = f"{base_pdf_name} - FullOutput.xlsx"

                if os.path.exists(excel_path):
                    with st.spinner("Loading saved Excel..."):
                        time.sleep(1.5)
                        sheets = pd.read_excel(excel_path, sheet_name=None)
                        first_sheet = list(sheets.keys())[0]
                        df_main = sheets[first_sheet]
                        st.markdown("**Main Fields:**")
                        st.dataframe(df_main, width="stretch")

                        for name, df in sheets.items():
                            if name != first_sheet:
                                st.markdown(f"**{name} Table:**")
                                st.dataframe(df, width="stretch")

                        st.download_button(
                            "⬇ Download Excel",
                            data=open(excel_path, "rb").read(),
                            file_name=f"{base_pdf_name} - Output.xlsx"
                        )
                else:
                    with st.spinner("Performing OCR & AI extraction..."):
                        try:
                            if not st.session_state.groq_client:
                                st.error("Groq client not available. Please reconnect.")
                            else:
                                text = pdf_to_text(temp_path)
                                doc_type = classify_doc(text, st.session_state.groq_client)
                                st.info(f"Detected document type: **{doc_type}**")

                                extracted_data = extract_fields(
                                    text, doc_type, SALESFORCE_COLUMNS, st.session_state.groq_client
                                )
                                cleaned_data = clean_salesforce_output(
                                    extracted_data, doc_type, SALESFORCE_COLUMNS
                                )

                                st.markdown("**Main Fields:**")
                                main_data = {k: v for k, v in cleaned_data.items() if not isinstance(v, list)}
                                df_main = pd.DataFrame(list(main_data.items()), columns=["Field", "Value"])
                                st.dataframe(df_main, width="stretch")

                                st.markdown("### Download Results")
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                                    df_main.to_excel(writer, index=False, sheet_name="Main")
                                st.download_button(
                                    "⬇ Download Excel",
                                    data=excel_buffer.getvalue(),
                                    file_name=f"{base_pdf_name} - Output.xlsx"
                                )
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

# ============================================================================
# Router: Login -> Groq -> App
# ============================================================================
if not st.session_state.authed:
    page_login()
elif not st.session_state.groq_valid:
    # If a valid key is already present in env/secrets, auto-validate once
    if PREFILL_GROQ and not st.session_state.groq_client:
        if validate_groq_key(PREFILL_GROQ):
            st.session_state.groq_valid = True
            st.rerun()
    page_groq_key()
else:
    page_main()
