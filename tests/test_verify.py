from datetime import date

from billsift.schema import Bill
from billsift.verify import (
    find_gstins_in_text,
    gstin_checksum_char,
    is_valid_gstin,
    normalize_date,
    repair_gstin,
    verify_bill,
)


def make_gstin(prefix14: str) -> str:
    return prefix14 + gstin_checksum_char(prefix14)


GOOD = make_gstin("27AABCU9603R1Z")  # Maharashtra-style test GSTIN


def good_bill(**kw) -> Bill:
    b = Bill(
        vendor_name="Test Traders",
        vendor_gstin=GOOD,
        invoice_number="INV-001",
        invoice_date="2026-09-01",
        taxable_amount=1000.0,
        gst_rate=18.0,
        cgst=90.0,
        sgst=90.0,
        total_amount=1180.0,
    )
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def test_known_public_gstin_checksum():
    # A widely used documentation example GSTIN
    assert is_valid_gstin("27AAPFU0939F1ZV")


def test_generated_gstin_valid_and_typo_invalid():
    assert is_valid_gstin(GOOD)
    typo = GOOD[:5] + ("A" if GOOD[5] != "A" else "B") + GOOD[6:]
    assert not is_valid_gstin(typo)
    assert not is_valid_gstin("")
    assert not is_valid_gstin("12345")


def test_clean_bill_has_no_flags():
    assert verify_bill(good_bill(), today=date(2026, 10, 1)) == []


def test_tax_mismatch_flagged():
    flags = verify_bill(good_bill(cgst=90.0, sgst=90.0, gst_rate=12.0), today=date(2026, 10, 1))
    assert any("Tax mismatch" in f for f in flags)


def test_total_mismatch_flagged():
    flags = verify_bill(good_bill(total_amount=1280.0), today=date(2026, 10, 1))
    assert any("Total mismatch" in f for f in flags)


def test_roundoff_tolerated():
    flags = verify_bill(good_bill(total_amount=1180.40), today=date(2026, 10, 1))
    assert flags == []


def test_cgst_sgst_must_match():
    flags = verify_bill(good_bill(cgst=100.0, sgst=80.0), today=date(2026, 10, 1))
    assert any("should be equal" in f for f in flags)


def test_igst_ok():
    b = good_bill(cgst=0.0, sgst=0.0, igst=180.0)
    assert verify_bill(b, today=date(2026, 10, 1)) == []


def test_future_date_flagged():
    flags = verify_bill(good_bill(invoice_date="2027-01-01"), today=date(2026, 10, 1))
    assert any("future" in f for f in flags)


def test_implied_rate_when_rate_missing():
    b = good_bill(gst_rate=0.0, cgst=70.0, sgst=70.0, total_amount=1140.0)
    flags = verify_bill(b, today=date(2026, 10, 1))
    assert any("Implied GST rate" in f for f in flags)


def test_grounding_catches_invented_number():
    text = f"Invoice INV-001 GSTIN {GOOD} Taxable 1,000.00 CGST 90.00 SGST 90.00 Total 1,180.00"
    assert verify_bill(good_bill(), ocr_text=text, today=date(2026, 10, 1)) == []
    bad = good_bill(taxable_amount=1100.0, cgst=99.0, sgst=99.0, total_amount=1298.0)
    flags = verify_bill(bad, ocr_text=text, today=date(2026, 10, 1))
    assert any("Not found in bill text" in f for f in flags)


def test_normalize_date_formats():
    assert normalize_date("05/09/2026") == "2026-09-05"
    assert normalize_date("5-9-26") == "2026-09-05"
    assert normalize_date("05 Sep 2026") == "2026-09-05"
    assert normalize_date("2026-09-05") == "2026-09-05"
    assert normalize_date("31/02/2026") == ""
    assert normalize_date("garbage") == ""


def test_gstin_repair_from_ocr_confusion():
    # OCR reads digit 0 as letter O inside the numeric block
    garbled = GOOD[:7] + GOOD[7:11].replace("9", "S", 1) + GOOD[11:] if "9" in GOOD[7:11] else GOOD
    text = f"GSTIN: {GOOD}"
    fixed, note = repair_gstin(garbled, text)
    assert fixed == GOOD
    assert find_gstins_in_text(text) == [GOOD]


def test_repair_leaves_valid_untouched():
    assert repair_gstin(GOOD, "") == (GOOD, None)
