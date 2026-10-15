"""BillSift: Streamlit UI.  Run with:  streamlit run app.py"""
from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from billsift.export import build_csv, build_workbook
from billsift.extract import DEFAULT_MODEL, OLLAMA_URL, list_ollama_models
from billsift.ocr import IMAGE_EXTS
from billsift.pipeline import analyze, is_problem, process_file
from billsift.schema import CATEGORIES, EDITABLE_FIELDS
from billsift.store import (
    backfill_category, connect, delete_bill, get_bill, list_bills, remember_vendor, row_to_bill,
    update_bill,
)
from billsift.verify import normalize_date, to_float

DATA_DIR = Path(os.environ.get("BILLSIFT_DATA", "data"))
TEXT_FIELDS = ["vendor_name", "vendor_gstin", "invoice_number", "invoice_date", "category"]
NUM_FIELDS = ["taxable_amount", "gst_rate", "cgst", "sgst", "igst", "total_amount"]

st.set_page_config(page_title="BillSift", page_icon="🧾", layout="wide")


@st.cache_resource
def get_conn():
    return connect(DATA_DIR / "billsift.db")


conn = get_conn()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🧾 BillSift")
    st.caption("Your bills never leave this computer.")
    models = list_ollama_models(OLLAMA_URL)
    if models is None:
        st.warning("Ollama is not running. Using rule-based extraction (less accurate). "
                   "Start Ollama for the AI model.")
        use_llm = False
        model = DEFAULT_MODEL
    else:
        use_llm = st.checkbox("Use local AI model", value=True)
        options = models or [DEFAULT_MODEL]
        default_ix = next((i for i, m in enumerate(options) if m.startswith(DEFAULT_MODEL.split(":")[0])), 0)
        model = st.selectbox("Model", options, index=default_ix)
        if not models:
            st.info(f"No models installed yet. Run:  ollama pull {DEFAULT_MODEL}")
    st.divider()
    st.caption("Bills with a ⚠️ need a quick human look: the checker found something that "
               "does not add up. Everything else passed all checks.")

# ------------------------------------------------------------------ upload
st.title("BillSift")
st.write("Drop in photos or PDFs of bills. A local AI reads them, then **plain code double-checks "
         "every GSTIN, tax amount and total** before anything reaches your spreadsheet.")

uploads = st.file_uploader("Bills (PNG, JPG, PDF)", type=["png", "jpg", "jpeg", "pdf"],
                           accept_multiple_files=True)
if uploads and st.button("Process bills", type="primary"):
    batch_dir = DATA_DIR / "uploads" / str(int(time.time() * 1000))
    batch_dir.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0.0, text="Starting...")
    errors = []
    for i, up in enumerate(uploads, 1):
        progress.progress((i - 1) / len(uploads), text=f"Reading {up.name} ({i}/{len(uploads)})")
        target = batch_dir / up.name
        target.write_bytes(up.getbuffer())
        try:
            process_file(conn, target, model=model, use_llm=use_llm)
        except Exception as e:  # keep going: one bad file should not stop the batch
            errors.append(f"{up.name}: {e}")
    progress.progress(1.0, text="Done")
    for err in errors:
        st.error(err)
    st.rerun()

# ------------------------------------------------------------------ table
rows = list_bills(conn)
if not rows:
    st.info("No bills yet. Upload a few above, or generate demo bills with "
            "`python scripts/make_samples.py` and upload them from the `samples` folder.")
    st.stop()

needs_review = [r for r in rows if r["status"] == "review"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bills", len(rows))
m2.metric("Need review", len(needs_review))
m3.metric("Taxable total", f"₹{sum(r['taxable_amount'] for r in rows):,.2f}")
m4.metric("GST total", f"₹{sum(r['cgst'] + r['sgst'] + r['igst'] for r in rows):,.2f}")

only_review = st.toggle("Show only bills that need review", value=False)
shown = needs_review if only_review else rows

df = pd.DataFrame([
    {
        "id": r["id"],
        "status": "⚠️ review" if r["status"] == "review" else "✅ ok",
        **{f: r[f] for f in EDITABLE_FIELDS},
        "issues": " | ".join(r["flags"]),
    }
    for r in shown
])
edited = st.data_editor(
    df,
    hide_index=True,
    width="stretch",
    key="editor",
    disabled=["id", "status", "issues"],
    column_config={
        "category": st.column_config.SelectboxColumn("category", options=CATEGORIES),
        **{f: st.column_config.NumberColumn(f, format="%.2f") for f in NUM_FIELDS},
        "issues": st.column_config.TextColumn("issues", width="large"),
    },
)

if st.button("💾 Save corrections"):
    changed = 0
    for _, new in edited.iterrows():
        old_row = get_bill(conn, int(new["id"]))
        if old_row is None:
            continue
        bill = row_to_bill(old_row)
        dirty = False
        for f in TEXT_FIELDS:
            val = "" if pd.isna(new[f]) else str(new[f]).strip()
            if f == "invoice_date" and val and val != bill.invoice_date:
                val = normalize_date(val) or val
            if val != (getattr(bill, f) or ""):
                setattr(bill, f, val)
                dirty = True
        for f in NUM_FIELDS:
            val = to_float(new[f])
            if abs(val - (getattr(bill, f) or 0.0)) > 1e-9:
                setattr(bill, f, val)
                dirty = True
        if dirty:
            # Re-check the corrected values. No grounding check: the human is the authority now.
            analyze(conn, bill, ocr_text=None, exclude_id=old_row["id"])
            update_bill(conn, old_row["id"], bill)
            if bill.category:
                remember_vendor(conn, bill)
                backfill_category(conn, bill)
            changed += 1
    if changed:
        st.success(f"Saved {changed} bill(s) and re-checked them.")
    st.rerun()

# ------------------------------------------------------------------ detail / export
left, right = st.columns([3, 2])
with left:
    with st.expander("🔍 Inspect a bill (original image, OCR text, issues)"):
        ids = [r["id"] for r in rows]
        sel = st.selectbox("Bill id", ids)
        row = get_bill(conn, sel)
        if row:
            stored = Path(row["stored_path"]) if row["stored_path"] else None
            c1, c2 = st.columns(2)
            with c1:
                if stored and stored.exists() and stored.suffix.lower() in IMAGE_EXTS:
                    st.image(str(stored), caption=row["source_file"])
                else:
                    st.caption(row["source_file"])
            with c2:
                for f in row["flags"]:
                    (st.warning if is_problem(f) else st.info)(f)
                if not row["flags"]:
                    st.success("Passed every check.")
                st.caption(f"Extracted with: {row['extractor']}")
                st.text_area("OCR text", row["ocr_text"], height=220)
            if st.button("🗑️ Delete this bill"):
                delete_bill(conn, sel)
                st.rerun()

with right:
    st.subheader("Export for your CA")
    stamp = date.today().isoformat()
    st.download_button(
        "⬇️ Excel (bills + GST summary + review list)", data=build_workbook(rows),
        file_name=f"billsift_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button("⬇️ CSV", data=build_csv(rows), file_name=f"billsift_{stamp}.csv",
                       mime="text/csv")
