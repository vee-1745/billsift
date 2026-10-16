"""Regression tests for the hard sample bills (samples/hard, made by scripts/make_hard_samples.py).

The invariant that matters: a bill that is NOT flagged must have every field correct.
Flagging a wrong extraction is fine; exporting a wrong number silently is not.
"""
import json
from pathlib import Path

import pytest

from billsift.ocr import fix_amount_separators, read_document
from billsift.pipeline import is_problem, process_file
from billsift.store import connect
from billsift.verify import normalize_date, verify_bill
from billsift.schema import Bill
from datetime import date

HARD = Path(__file__).resolve().parent.parent / "samples" / "hard"
FIELDS = ["vendor_gstin", "invoice_number", "invoice_date", "taxable_amount", "total_amount"]


def same(field, got, want):
    if field in ("taxable_amount", "total_amount"):
        return abs(float(got) - float(want)) < 0.011
    return str(got).strip().upper() == str(want).strip().upper()


@pytest.mark.skipif(not (HARD / "manifest.json").exists(), reason="hard samples not generated")
def test_no_silent_errors_on_hard_samples():
    manifest = json.loads((HARD / "manifest.json").read_text())
    conn = connect(":memory:")
    silent = []
    for name, truth in manifest.items():
        _, bill = process_file(conn, HARD / name, use_llm=False)
        flagged = any(is_problem(f) for f in bill.flags)
        wrong = [f for f in FIELDS if not same(f, getattr(bill, f), truth[f])]
        if truth["should_flag"]:
            assert flagged, f"{name} should have been flagged"
        elif wrong and not flagged:
            silent.append((name, wrong))
    assert silent == [], f"wrong values passed unflagged: {silent}"


def test_yyyy_mm_dd_date_is_not_misread():
    import billsift.extract as ex

    r = ex.heuristic_extract("Invoice No: X1  Date: 2026/10/11\nTotal 10.00")
    assert normalize_date(r["invoice_date"]) == "2026-10-11"


def test_inv_dot_no_label():
    import billsift.extract as ex

    assert ex.heuristic_extract("Inv. No.: AB/25-26/0098  Date: 12-Oct-26")["invoice_number"] == "AB/25-26/0098"


def test_ocr_mangled_payable_label():
    import billsift.extract as ex

    assert ex.heuristic_extract("Taxable Value 1,140.00\nAmount Payabbe 1,276.80")["total_amount"] == 1276.8


def test_dot_as_thousands_separator_fixed():
    assert fix_amount_separators("Grand Total  3.339.40") == "Grand Total  3,339.40"
    assert fix_amount_separators("Rate 1.450.00 and 25.50") == "Rate 1,450.00 and 25.50"
    assert fix_amount_separators("Qty 3.5L  60.00") == "Qty 3.5L  60.00"


def test_stale_year_is_flagged():
    b = Bill(vendor_name="X", vendor_gstin="", invoice_number="1", invoice_date="2011-10-26",
             taxable_amount=100, gst_rate=0, total_amount=100)
    assert any("more than 3 years old" in f for f in verify_bill(b, today=date(2026, 10, 17)))


def test_digital_pdf_label_and_value_in_separate_cells_end_up_on_one_line(tmp_path):
    import pymupdf

    pdf = tmp_path / "cells.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Tax Invoice  Seller GSTIN  27AAHCT9988K1Z8", fontsize=11)
    for y, label, val in [(140, "Taxable Value", "1,000.00"), (160, "IGST @ 18%", "180.00"), (180, "TOTAL", "1,180.00")]:
        page.insert_text((50, y), label, fontsize=11)   # separate cells, like a real invoice table
        page.insert_text((400, y), val, fontsize=11)
    doc.save(pdf)
    lines = read_document(pdf).splitlines()
    assert any("Taxable Value" in ln and "1,000.00" in ln for ln in lines)
    assert any("TOTAL" in ln and "1,180.00" in ln for ln in lines)
