"""Turn OCR text into a structured Bill.

Primary path: a small open-weight LLM served locally by Ollama, constrained to a JSON schema.
Fallback path: deterministic rules (regex), used when Ollama is not running. The rules
extractor is also used to cross-check the LLM: if the model returns a number that is not
actually in the bill text, a grounded value from the rules extractor replaces it.
"""
from __future__ import annotations

import json
import re

import requests

from .schema import BILL_JSON_SCHEMA, Bill
from .verify import (
    find_gstins_in_text,
    normalize_date,
    numbers_in_text,
    repair_gstin,
    to_float,
)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """You extract fields from the OCR text of an Indian GST tax invoice or bill.
Rules:
- Copy values exactly as printed. Never calculate, round or guess.
- vendor_name and vendor_gstin belong to the SELLER who issued the bill, not the buyer.
- Amounts are plain numbers without currency symbols or commas.
- gst_rate is the TOTAL GST percentage: CGST 9% + SGST 9% means 18, CGST 2.5% + SGST 2.5% means 5.
  Never return only the CGST or SGST percentage.
- If a field is not present, use "" for text and 0 for numbers.
- invoice_date: copy the date exactly as printed."""


class ExtractionError(RuntimeError):
    pass


# ---------------------------------------------------------------- Ollama
def list_ollama_models(url: str = OLLAMA_URL, timeout: float = 2.0) -> list[str] | None:
    """Model names installed in Ollama, or None if the server is not reachable."""
    try:
        r = requests.get(f"{url}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None


def llm_extract(ocr_text: str, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
                timeout: float = 180.0) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": BILL_JSON_SCHEMA,
        # keep_alive: stay loaded between bills (reloading the model is a big part of the wait).
        # num_predict: the JSON answer is short, so cap it to stop any rambling generation.
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": 350},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Bill text:\n\"\"\"\n{ocr_text[:6000]}\n\"\"\""},
        ],
    }
    try:
        r = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except (requests.RequestException, ValueError, KeyError) as e:
        raise ExtractionError(f"Ollama extraction failed: {e}") from e


# ---------------------------------------------------------------- rules extractor
_NUM = re.compile(r"\d[\d,]*\.\d{1,2}|\d[\d,]*")


def _last_number(line: str) -> float:
    nums = _NUM.findall(line)
    return to_float(nums[-1]) if nums else 0.0


def _find_line_amount(lines: list[str], patterns: list[str], exclude: str = "") -> float:
    """Amount from the last line matching the first pattern (in priority order) that hits."""
    for pat in patterns:
        hits = [
            ln for ln in lines
            if re.search(pat, ln, re.I) and not (exclude and re.search(exclude, ln, re.I))
        ]
        if hits:
            val = _last_number(hits[-1])
            if val:
                return val
    return 0.0


def _rate_from_line(lines: list[str], label: str) -> float:
    for ln in lines:
        if re.search(label, ln, re.I):
            m = re.search(r"@?\s*(\d+(?:\.\d+)?)\s*%", ln)
            if m:
                return float(m.group(1))
    return 0.0


def heuristic_extract(text: str) -> dict:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: dict = {k: "" for k in ("vendor_name", "vendor_gstin", "invoice_number", "invoice_date")}
    out.update({k: 0.0 for k in ("taxable_amount", "cgst", "sgst", "igst", "gst_rate", "total_amount")})

    for ln in lines:
        if not re.search(r"tax\s*invoice|gstin|invoice|bill|receipt", ln, re.I):
            out["vendor_name"] = ln
            break

    m = re.search(r"GSTIN\W{0,4}([A-Z0-9]{15})", text or "", re.I)
    if m:
        out["vendor_gstin"] = m.group(1).upper()
    else:
        valid = find_gstins_in_text(text)
        out["vendor_gstin"] = valid[0] if valid else ""

    m = re.search(r"(?:Invoice|Inv\.?|Bill)\s*(?:No|Number|Num|#)\.?\s*[:.\-]?\s*([A-Za-z0-9][A-Za-z0-9/\-_.]*)",
                  text or "", re.I)
    if m:
        out["invoice_number"] = m.group(1).rstrip(".")

    # (?<!\d) stops a match starting mid-number: "2026/10/11" must not be read as "26/10/11".
    date_pat = (r"((?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
                r"|\d{1,2}[-/. ](?:\d{1,2}|[A-Za-z015]{3,9})[-/. ,]+\d{2,4}(?!\d)))")
    m = re.search(r"(?:Date|Dated|Dt)\s*[:.\-]?\s*" + date_pat, text or "", re.I) or re.search(date_pat, text or "")
    if m:
        out["invoice_date"] = m.group(1)

    out["taxable_amount"] = _find_line_amount(lines, [r"taxable", r"sub\s*-?\s*total"])
    out["cgst"] = _find_line_amount(lines, [r"\bcgst\b"])
    out["sgst"] = _find_line_amount(lines, [r"\bsgst\b", r"\butgst\b"])
    out["igst"] = _find_line_amount(lines, [r"\bigst\b"])
    out["total_amount"] = _find_line_amount(
        lines,
        # [l1!|I] tolerates OCR reading the final "l" of "Total" as "!" or "1"
        [r"grand\s*tota[l1!|I]", r"tota[l1!|I]\s*amount", r"net\s*(?:amount|payable)",
         r"amount\s*payab\w*", r"\btota[l1!|I]"],  # payab\w*: OCR turns "Payable" into "Payabbe"
        exclude=r"taxable|sub\s*-?\s*tota|tax\s*tota",
    )
    rate = _rate_from_line(lines, r"\bigst\b")
    if not rate:
        half = _rate_from_line(lines, r"\bcgst\b")
        rate = half * 2 if half else 0.0
    out["gst_rate"] = rate
    return out


# ---------------------------------------------------------------- assemble a Bill
_NUMERIC = ("taxable_amount", "cgst", "sgst", "igst", "gst_rate", "total_amount")
_AMOUNTS = ("taxable_amount", "cgst", "sgst", "igst", "total_amount")


def extract_bill(ocr_text: str, source_file: str = "", model: str = DEFAULT_MODEL,
                 use_llm: bool = True, url: str = OLLAMA_URL) -> tuple[Bill, list[str]]:
    """Returns (bill, notes). Notes are informational messages about automatic repairs."""
    notes: list[str] = []
    rules = heuristic_extract(ocr_text)
    raw = rules
    extractor = "rules"

    if use_llm:
        try:
            raw = llm_extract(ocr_text, model=model, url=url)
            extractor = "llm"
        except ExtractionError as e:
            notes.append(f"{e}; fell back to rule-based extraction")
            raw = rules

    bill = Bill(source_file=source_file, extractor=extractor)
    bill.vendor_name = str(raw.get("vendor_name") or rules["vendor_name"]).strip()
    bill.invoice_number = str(raw.get("invoice_number") or rules["invoice_number"]).strip()
    bill.invoice_date = normalize_date(str(raw.get("invoice_date") or "")) or normalize_date(rules["invoice_date"])
    for f in _NUMERIC:
        setattr(bill, f, to_float(raw.get(f)))

    # Cross-check the model against the text: replace ungrounded amounts with grounded rule values.
    if extractor == "llm":
        present = numbers_in_text(ocr_text)
        for f in _AMOUNTS:
            val, alt = getattr(bill, f), rules[f]
            if val > 0 and round(val, 2) not in present and alt > 0 and round(alt, 2) in present:
                notes.append(f"{f}: model said {val:.2f}, not in bill text; used {alt:.2f} from the text")
                setattr(bill, f, alt)
            elif val == 0 and alt > 0:
                setattr(bill, f, alt)
        if bill.gst_rate == 0 and rules["gst_rate"]:
            bill.gst_rate = rules["gst_rate"]

    gstin, note = repair_gstin(str(raw.get("vendor_gstin") or rules["vendor_gstin"]), ocr_text)
    bill.vendor_gstin = gstin
    if note:
        notes.append(note)
    return bill, notes
