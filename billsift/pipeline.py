"""Glue: file -> OCR -> extraction -> verification -> database."""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from .extract import DEFAULT_MODEL, OLLAMA_URL, extract_bill
from .ocr import read_document
from .schema import Bill
from .store import find_duplicate, insert_bill, lookup_category
from .verify import normalize_gst_rate, verify_bill

NOTE_PREFIX = "Note: "


def is_problem(flag: str) -> bool:
    return not flag.startswith(NOTE_PREFIX)


def analyze(conn: sqlite3.Connection, bill: Bill, ocr_text: str | None = None,
            exclude_id: int | None = None, notes: list[str] | None = None) -> Bill:
    """Run every check on a bill and set its flags and status."""
    notes = list(notes or [])
    rate_note = normalize_gst_rate(bill)
    if rate_note:
        notes.append(rate_note)
    flags = verify_bill(bill, ocr_text)
    dup = find_duplicate(conn, bill, exclude_id)
    if dup:
        flags.append(f"Possible duplicate of bill #{dup} (same vendor and invoice number)")
    problems = list(flags)
    bill.flags = [NOTE_PREFIX + n for n in notes] + problems
    bill.status = "review" if problems else "ok"
    return bill


def process_file(conn: sqlite3.Connection, path: str | Path, *, model: str = DEFAULT_MODEL,
                 use_llm: bool = True, url: str = OLLAMA_URL,
                 uploads_dir: str | Path | None = None) -> tuple[int, Bill]:
    path = Path(path)
    stored = path
    if uploads_dir:
        Path(uploads_dir).mkdir(parents=True, exist_ok=True)
        stored = Path(uploads_dir) / path.name
        if path.resolve() != stored.resolve():
            shutil.copyfile(path, stored)

    text = read_document(stored)
    bill, notes = extract_bill(text, source_file=path.name, model=model, use_llm=use_llm, url=url)
    if not bill.category:
        bill.category = lookup_category(conn, bill.vendor_gstin, bill.vendor_name)
    analyze(conn, bill, ocr_text=text, notes=notes)
    bill_id = insert_bill(conn, bill, ocr_text=text, stored_path=str(stored))
    return bill_id, bill
