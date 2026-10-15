"""Extraction tests. The LLM is mocked so these run without Ollama."""
from pathlib import Path

import pytest

import billsift.extract as ex
from billsift.verify import verify_bill

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

TEXT = """Shree Balaji Fuels
GSTIN: 27AABCP1234A1Z4
TAX INVOICE
Invoice No: BF/2026/0412  Date: 03/10/2026
Item  Qty  Rate  Amount
Petrol (Premium)  20  105.50  2,110.00
Taxable Amount  2,110.00
CGST @ 9%  189.90
SGST @ 9%  189.90
Grand Total  2,489.80
"""


def test_rules_extractor_reads_clean_bill():
    r = ex.heuristic_extract(TEXT)
    assert r["vendor_name"] == "Shree Balaji Fuels"
    assert r["invoice_number"] == "BF/2026/0412"
    assert r["taxable_amount"] == 2110.0
    assert r["cgst"] == r["sgst"] == 189.9
    assert r["total_amount"] == 2489.8
    assert r["gst_rate"] == 18.0


def test_pipeline_rules_only_gives_clean_flags():
    bill, _ = ex.extract_bill(TEXT, "x.png", use_llm=False)
    assert bill.invoice_date == "2026-10-03"
    assert verify_bill(bill, TEXT) == []


def test_llm_hallucination_is_replaced_by_grounded_value(monkeypatch):
    wrong = {
        "vendor_name": "Shree Balaji Fuels", "vendor_gstin": "27AABCP1234A1Z4",
        "invoice_number": "BF/2026/0412", "invoice_date": "03/10/2026",
        "taxable_amount": 2110.0, "cgst": 189.9, "sgst": 189.9, "igst": 0,
        "gst_rate": 18, "total_amount": 2498.80,  # digits swapped by the "model"
    }
    monkeypatch.setattr(ex, "llm_extract", lambda *a, **k: wrong)
    bill, notes = ex.extract_bill(TEXT, "x.png", use_llm=True)
    assert bill.total_amount == 2489.8
    assert any("not in bill text" in n for n in notes)


def test_llm_failure_falls_back_to_rules(monkeypatch):
    def boom(*a, **k):
        raise ex.ExtractionError("no server")

    monkeypatch.setattr(ex, "llm_extract", boom)
    bill, notes = ex.extract_bill(TEXT, "x.png", use_llm=True)
    assert bill.extractor == "rules"
    assert bill.total_amount == 2489.8
    assert any("fell back" in n for n in notes)


@pytest.mark.skipif(not SAMPLES.exists(), reason="samples not generated")
def test_end_to_end_on_sample_images():
    from billsift.ocr import read_document

    def run(name):
        text = read_document(SAMPLES / name)
        bill, _ = ex.extract_bill(text, name, use_llm=False)
        return bill, verify_bill(bill, text)

    for ok in ["01_petrol_pump_ok.png", "02_cafe_ok.png", "03_stationery_igst_ok.png", "04_software_ok.png"]:
        _, flags = run(ok)
        assert flags == [], (ok, flags)

    _, flags = run("05_BAD_total_tampered.png")
    assert any("Total mismatch" in f for f in flags)
    _, flags = run("06_BAD_gstin_typo.png")
    assert any("Invalid GSTIN" in f for f in flags)


def test_pipeline_fixes_half_rate_from_llm(monkeypatch):
    from billsift.pipeline import analyze
    from billsift.store import connect

    text = ("Cafe Chaicoffee\nGSTIN: 27AAECC5678B1ZO\nTAX INVOICE\nInvoice No: CC-7781  Date: 05-10-2026\n"
            "Taxable Amount  660.00\nCGST @ 2.5%  16.50\nSGST @ 2.5%  16.50\nGrand Total  693.00\n")
    llm = {"vendor_name": "Cafe Chaicoffee", "vendor_gstin": "27AAECC5678B1ZO",
           "invoice_number": "CC-7781", "invoice_date": "05-10-2026", "taxable_amount": 660.0,
           "cgst": 16.5, "sgst": 16.5, "igst": 0, "gst_rate": 2.5, "total_amount": 693.0}
    monkeypatch.setattr(ex, "llm_extract", lambda *a, **k: llm)
    bill, notes = ex.extract_bill(text, "cafe.png", use_llm=True)
    analyze(connect(":memory:"), bill, ocr_text=text, notes=notes)
    assert bill.gst_rate == 5.0 and bill.status == "ok"
    assert any(f.startswith("Note:") and "half" in f for f in bill.flags)
