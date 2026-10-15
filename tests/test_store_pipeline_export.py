import io
from pathlib import Path

import pandas as pd
import pytest

from billsift.export import build_csv, build_workbook, financial_year, quarter
from billsift.pipeline import analyze, is_problem, process_file
from billsift.schema import Bill
from billsift.store import (
    connect, find_duplicate, get_bill, insert_bill, list_bills, lookup_category,
    remember_vendor, row_to_bill, update_bill,
)
from billsift.verify import gstin_checksum_char

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
GST = "27AABCP1234A1Z" + gstin_checksum_char("27AABCP1234A1Z")


def bill(**kw):
    b = Bill(vendor_name="Shree Balaji Fuels", vendor_gstin=GST, invoice_number="BF/1",
             invoice_date="2026-10-03", taxable_amount=1000, gst_rate=18, cgst=90, sgst=90,
             total_amount=1180)
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def test_insert_roundtrip_and_update():
    conn = connect(":memory:")
    b = analyze(conn, bill())
    bid = insert_bill(conn, b, "text")
    row = get_bill(conn, bid)
    assert row["vendor_name"] == "Shree Balaji Fuels" and row["status"] == "ok"
    b2 = row_to_bill(row)
    b2.category = "Fuel"
    update_bill(conn, bid, b2)
    assert get_bill(conn, bid)["category"] == "Fuel"


def test_duplicate_detection_by_gstin_and_by_name():
    conn = connect(":memory:")
    insert_bill(conn, analyze(conn, bill()), "")
    assert find_duplicate(conn, bill()) is not None
    assert find_duplicate(conn, bill(invoice_number="bf-1")) is not None  # formatting-insensitive
    assert find_duplicate(conn, bill(invoice_number="BF/2")) is None
    # bad GSTIN but same vendor name + invoice number is still caught
    assert find_duplicate(conn, bill(vendor_gstin="27AAXXXX")) is not None
    flagged = analyze(conn, bill())
    assert any("duplicate" in f.lower() for f in flagged.flags)


def test_vendor_memory_by_gstin_and_name():
    conn = connect(":memory:")
    b = bill(category="Fuel")
    remember_vendor(conn, b)
    assert lookup_category(conn, GST, "whatever") == "Fuel"
    assert lookup_category(conn, "BADGSTIN", "Shree  Balaji FUELS") == "Fuel"
    assert lookup_category(conn, "", "Unknown Vendor") == ""


def test_export_workbook_and_summary():
    conn = connect(":memory:")
    for i, total in enumerate([1180.0, 2360.0]):
        b = bill(invoice_number=f"INV{i}", taxable_amount=total / 1.18,
                 cgst=round(total / 1.18 * 0.09, 2), sgst=round(total / 1.18 * 0.09, 2),
                 total_amount=total, category="Fuel")
        insert_bill(conn, analyze(conn, b), "")
    rows = list_bills(conn)
    xlsx = build_workbook(rows)
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None)
    assert set(sheets) == {"Bills", "GST Summary", "By Category", "Needs Review"}
    assert len(sheets["Bills"]) == 2
    assert sheets["GST Summary"]["bills"].sum() == 2
    assert abs(sheets["By Category"]["total_amount"].sum() - 3540.0) < 0.01
    assert b"vendor_name" in build_csv(rows)


def test_financial_year_and_quarter():
    assert financial_year("2026-10-03") == "FY 2026-27"
    assert financial_year("2027-02-10") == "FY 2026-27"
    assert quarter("2026-10-03") == "Q3" and quarter("2027-02-10") == "Q4"
    assert financial_year("") == "Unknown"


@pytest.mark.skipif(not SAMPLES.exists(), reason="samples not generated")
def test_full_pipeline_on_sample_bills(tmp_path):
    conn = connect(tmp_path / "t.db")
    results = {}
    for f in sorted(SAMPLES.glob("*.png")):
        bid, b = process_file(conn, f, use_llm=False)
        if f.name.startswith("01"):
            b1 = row_to_bill(get_bill(conn, bid))
            b1.category = "Fuel"
            remember_vendor(conn, b1)  # the user labels the first petrol bill once
        results[f.name] = b

    for name in ["01_petrol_pump_ok.png", "02_cafe_ok.png",
                 "03_stationery_igst_ok.png", "04_software_ok.png"]:
        assert results[name].status == "ok", (name, results[name].flags)
    assert any("Total mismatch" in f for f in results["05_BAD_total_tampered.png"].flags)
    assert any("Invalid GSTIN" in f for f in results["06_BAD_gstin_typo.png"].flags)
    assert any("duplicate" in f.lower() for f in results["07_duplicate_of_01.png"].flags)
    # vendor memory: later bills from the same vendor are categorised automatically
    assert results["06_BAD_gstin_typo.png"].category == "Fuel"
    assert results["07_duplicate_of_01.png"].category == "Fuel"
    assert all(is_problem(f) for f in results["05_BAD_total_tampered.png"].flags)


def test_backfill_category_updates_older_uncategorised_bills():
    from billsift.store import backfill_category

    conn = connect(":memory:")
    a = insert_bill(conn, analyze(conn, bill(invoice_number="A1")), "")
    b = insert_bill(conn, analyze(conn, bill(invoice_number="A2")), "")
    c = insert_bill(conn, analyze(conn, bill(invoice_number="B1", vendor_name="Other Co",
                                             vendor_gstin="")), "")
    n = backfill_category(conn, bill(category="Fuel"))
    assert n == 2
    assert get_bill(conn, a)["category"] == get_bill(conn, b)["category"] == "Fuel"
    assert get_bill(conn, c)["category"] == ""
