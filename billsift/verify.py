"""The verification layer: plain code that checks what the LLM extracted.

Small local models make small mistakes (a wrong digit, a swapped field). In accounting a
silently wrong digit is worse than a missing one, so nothing is trusted until it passes
these checks. Anything that fails is flagged for a human instead of exported silently.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from .schema import Bill

_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(
    r"^(0[1-9]|[12][0-9]|3[0-8]|97|99)[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$"
)
STANDARD_RATES = [0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0]

# Characters OCR commonly confuses, by what the position expects.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
_LOOKALIKE = {**_TO_LETTER, **{v: k for k, v in _TO_LETTER.items()}}
_FOLD = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def _fold(s: str) -> str:
    """Collapse OCR look-alike characters so text can be compared tolerantly."""
    return s.translate(_FOLD)


# ---------------------------------------------------------------- GSTIN
def gstin_checksum_char(first14: str) -> str:
    """Official GSTIN check character (mod-36, alternating weights 1,2)."""
    total = 0
    for i, ch in enumerate(first14):
        v = _CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += v // 36 + v % 36
    return _CHARS[(36 - total % 36) % 36]


def is_valid_gstin(gstin: str) -> bool:
    g = (gstin or "").strip().upper()
    if not GSTIN_RE.match(g):
        return False
    return gstin_checksum_char(g[:14]) == g[14]


def repair_gstin_token(token: str) -> str | None:
    """Try to turn an OCR-garbled 15-char token into a valid GSTIN by fixing look-alike chars."""
    t = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    if len(t) != 15:
        return None
    if is_valid_gstin(t):
        return t
    out = []
    for i, ch in enumerate(t):
        if i in (0, 1, 7, 8, 9, 10):  # digit positions
            out.append(_TO_DIGIT.get(ch, ch))
        elif i in (2, 3, 4, 5, 6, 11):  # letter positions
            out.append(_TO_LETTER.get(ch, ch))
        elif i == 13:
            out.append("Z")
        else:
            out.append(ch)
    cand = "".join(out)
    if is_valid_gstin(cand):
        return cand
    # The last character is the checksum and may legitimately be a letter or a digit, so OCR
    # can swap look-alikes there (O/0, I/1, S/5...). Accept only if the checksum confirms it.
    if all(c in _CHARS for c in cand[:14]):
        expected = gstin_checksum_char(cand[:14])
        if expected in (cand[14], _LOOKALIKE.get(cand[14])):
            fixed = cand[:14] + expected
            if is_valid_gstin(fixed):
                return fixed
    return None


def find_gstins_in_text(text: str) -> list[str]:
    """All checksum-valid GSTINs found in OCR text (after cleaning OCR confusions)."""
    found: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]{15}", text or ""):
        g = repair_gstin_token(token)
        if g and g not in found:
            found.append(g)
    return found


def _edit_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def repair_gstin(llm_value: str, ocr_text: str) -> tuple[str, str | None]:
    """Return (gstin, note). Keep a valid value; otherwise repair from the OCR text if safe."""
    v = re.sub(r"[^A-Za-z0-9]", "", llm_value or "").upper()
    if is_valid_gstin(v):
        return v, None
    candidates = find_gstins_in_text(ocr_text)
    if v and candidates:
        best = min(candidates, key=lambda c: _edit_distance(v, c))
        if _edit_distance(v, best) <= 3:
            return best, f"GSTIN repaired from OCR text ({v} -> {best})"
    if not v and len(candidates) == 1:
        return candidates[0], "GSTIN taken from OCR text (model returned none)"
    return v, None


# ---------------------------------------------------------------- dates
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def normalize_date(s: str) -> str:
    """Parse common Indian bill date formats (day-first) to ISO. Returns '' if unparseable."""
    s = (s or "").strip()
    if not s:
        return ""
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.match(r"^(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})$", s)
        if m:
            d, mo, y = map(int, m.groups())
        else:
            m = re.match(r"^(\d{1,2})[-/. ]([A-Za-z015]{3,9})[-/. ,]*(\d{2,4})$", s)
            if not m:
                return ""
            d = int(m.group(1))
            # OCR often reads the o in "Oct" as a zero, "Sep" as "5ep", "Jul" as "Ju1"
            month_word = m.group(2).lower().translate(str.maketrans("015", "ols"))
            mo = _MONTHS.get(month_word[:3], 0)
            y = int(m.group(3))
    if y < 100:
        y += 2000
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return ""


# ---------------------------------------------------------------- numbers / grounding
def to_float(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^\d.\-]", "", str(x or ""))
    try:
        return float(s)
    except ValueError:
        return 0.0


def numbers_in_text(text: str) -> set[float]:
    """Every number that literally appears in the OCR text, commas stripped."""
    nums: set[float] = set()
    for tok in re.findall(r"\d[\d,]*\.?\d*", text or ""):
        try:
            nums.add(round(float(tok.replace(",", "")), 2))
        except ValueError:
            pass
    return nums


def normalize_gst_rate(bill: Bill) -> str | None:
    """Models often return the per-component rate (CGST 2.5%) instead of the combined one (5%).

    If the stated rate does not explain the tax amounts, but exactly doubling it does and the
    result is a standard GST slab, use the doubled rate and return a note. A rate that is
    simply wrong (neither explains the amounts) is left alone, so the checker still flags it.
    """
    tax = bill.cgst + bill.sgst
    if bill.taxable_amount <= 0 or bill.cgst <= 0 or bill.sgst <= 0 or bill.gst_rate <= 0:
        return None
    expected = bill.taxable_amount * bill.gst_rate / 100
    if abs(expected - tax) <= max(1.0, 0.005 * expected):
        return None  # stated rate already matches
    doubled = bill.gst_rate * 2
    implied = tax / bill.taxable_amount * 100
    if any(abs(doubled - r) < 0.01 for r in STANDARD_RATES) and abs(implied - doubled) <= 0.3:
        old = bill.gst_rate
        bill.gst_rate = doubled
        return f"GST rate {old:g}% looked like the CGST/SGST half; using combined {doubled:g}%"
    return None


# ---------------------------------------------------------------- main check
def verify_bill(bill: Bill, ocr_text: str | None = None, today: date | None = None) -> list[str]:
    """Run every check. Returns human-readable flags (empty list = clean).

    If ocr_text is given, a grounding check confirms the model did not invent values:
    every amount, the GSTIN and the invoice number must literally appear in the OCR text.
    """
    flags: list[str] = []
    today = today or date.today()

    # 1. GSTIN
    if not bill.vendor_gstin:
        flags.append("Missing vendor GSTIN")
    elif not is_valid_gstin(bill.vendor_gstin):
        flags.append(f"Invalid GSTIN ({bill.vendor_gstin}): format or checksum fails")

    # 2. Date
    if not bill.invoice_date:
        flags.append("Missing or unreadable invoice date")
    else:
        try:
            d = datetime.strptime(bill.invoice_date, "%Y-%m-%d").date()
            if d > today:
                flags.append(f"Invoice date {bill.invoice_date} is in the future")
        except ValueError:
            flags.append(f"Invoice date not in a recognised format ({bill.invoice_date})")

    if not bill.invoice_number:
        flags.append("Missing invoice number")

    # 3. Tax structure
    tax = round(bill.cgst + bill.sgst + bill.igst, 2)
    if bill.igst > 0 and (bill.cgst > 0 or bill.sgst > 0):
        flags.append("Both IGST and CGST/SGST present (should be one or the other)")
    if bill.cgst > 0 or bill.sgst > 0:
        if abs(bill.cgst - bill.sgst) > 0.011:
            flags.append(f"CGST ({bill.cgst}) and SGST ({bill.sgst}) should be equal")

    # 4. Tax maths
    if bill.taxable_amount <= 0:
        flags.append("Missing taxable amount")
    else:
        if bill.gst_rate > 0:
            if all(abs(bill.gst_rate - r) > 0.01 for r in STANDARD_RATES):
                flags.append(f"Unusual GST rate {bill.gst_rate}% (not a standard slab)")
            expected = bill.taxable_amount * bill.gst_rate / 100
            if abs(expected - tax) > max(1.0, 0.005 * expected):
                flags.append(
                    f"Tax mismatch: {bill.gst_rate}% of {bill.taxable_amount:.2f} is "
                    f"{expected:.2f}, but bill shows {tax:.2f}"
                )
        elif tax > 0:
            implied = tax / bill.taxable_amount * 100
            if min(abs(implied - r) for r in STANDARD_RATES) > 0.3:
                flags.append(f"Implied GST rate {implied:.2f}% is not a standard slab")

    # 5. Total
    if bill.total_amount <= 0:
        flags.append("Missing total amount")
    elif bill.taxable_amount > 0:
        expected_total = bill.taxable_amount + tax
        if abs(expected_total - bill.total_amount) > 1.0:
            flags.append(
                f"Total mismatch: taxable + tax = {expected_total:.2f}, bill total = "
                f"{bill.total_amount:.2f}"
            )

    # 6. Grounding (only when we still have the OCR text)
    if ocr_text:
        nums = numbers_in_text(ocr_text)
        for label, val in [
            ("taxable amount", bill.taxable_amount),
            ("CGST", bill.cgst),
            ("SGST", bill.sgst),
            ("IGST", bill.igst),
            ("total", bill.total_amount),
        ]:
            if val > 0 and round(val, 2) not in nums:
                flags.append(f"Not found in bill text: {label} {val:.2f} (possible misread)")
        # Compare with look-alike characters (O/0, I/1, S/5...) folded together, so a repaired
        # GSTIN is still recognised in text where OCR printed the look-alike.
        compact = _fold(re.sub(r"[^A-Za-z0-9]", "", ocr_text).upper())
        if bill.vendor_gstin and _fold(bill.vendor_gstin.upper()) not in compact:
            flags.append("GSTIN not found in bill text")
        if bill.invoice_number:
            inv = _fold(re.sub(r"[^A-Za-z0-9]", "", bill.invoice_number).upper())
            if inv and inv not in compact:
                flags.append("Invoice number not found in bill text")

    return flags
