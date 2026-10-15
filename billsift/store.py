"""SQLite storage, duplicate detection and per-vendor category memory.

Vendor memory: when you correct a bill's category once, every later bill from that vendor
gets the same category automatically. No fine-tuning needed, and it stays on your laptop.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from .schema import Bill
from .verify import is_valid_gstin

SCHEMA = """
CREATE TABLE IF NOT EXISTS bills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT, stored_path TEXT,
    vendor_name TEXT, vendor_gstin TEXT, invoice_number TEXT, invoice_date TEXT,
    taxable_amount REAL, gst_rate REAL, cgst REAL, sgst REAL, igst REAL, total_amount REAL,
    category TEXT, extractor TEXT, flags TEXT, status TEXT, ocr_text TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS vendor_rules (
    key TEXT PRIMARY KEY, vendor_name TEXT, category TEXT, updated_at TEXT
);
"""

BILL_COLUMNS = [
    "source_file", "vendor_name", "vendor_gstin", "invoice_number", "invoice_date",
    "taxable_amount", "gst_rate", "cgst", "sgst", "igst", "total_amount",
    "category", "extractor", "status",
]


def connect(path: str | Path = "data/billsift.db") -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def vendor_keys(gstin: str, name: str) -> list[str]:
    keys = []
    if is_valid_gstin(gstin):
        keys.append("gstin:" + _norm(gstin))
    if _norm(name):
        keys.append("name:" + _norm(name))
    return keys


# ---------------------------------------------------------------- bills
def insert_bill(conn: sqlite3.Connection, bill: Bill, ocr_text: str = "", stored_path: str = "") -> int:
    cols = BILL_COLUMNS + ["flags", "ocr_text", "stored_path", "created_at"]
    vals = [getattr(bill, c) for c in BILL_COLUMNS] + [
        json.dumps(bill.flags), ocr_text, stored_path, datetime.now().isoformat(timespec="seconds"),
    ]
    cur = conn.execute(
        f"INSERT INTO bills ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals
    )
    conn.commit()
    return cur.lastrowid


def update_bill(conn: sqlite3.Connection, bill_id: int, bill: Bill) -> None:
    sets = BILL_COLUMNS + ["flags"]
    vals = [getattr(bill, c) for c in BILL_COLUMNS] + [json.dumps(bill.flags)]
    conn.execute(
        f"UPDATE bills SET {', '.join(c + '=?' for c in sets)} WHERE id=?", vals + [bill_id]
    )
    conn.commit()


def delete_bill(conn: sqlite3.Connection, bill_id: int) -> None:
    conn.execute("DELETE FROM bills WHERE id=?", (bill_id,))
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["flags"] = json.loads(d.get("flags") or "[]")
    return d


def list_bills(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM bills ORDER BY id DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_bill(conn: sqlite3.Connection, bill_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM bills WHERE id=?", (bill_id,)).fetchone()
    return _row_to_dict(row) if row else None


def row_to_bill(row: dict) -> Bill:
    return Bill(**{c: row[c] for c in BILL_COLUMNS}, flags=list(row.get("flags") or []))


# ---------------------------------------------------------------- duplicates
def find_duplicate(conn: sqlite3.Connection, bill: Bill, exclude_id: int | None = None) -> int | None:
    """Same vendor (GSTIN or name) and same invoice number already stored?"""
    inv = _norm(bill.invoice_number)
    if not inv:
        return None
    for r in conn.execute("SELECT id, vendor_gstin, vendor_name, invoice_number FROM bills"):
        if exclude_id is not None and r["id"] == exclude_id:
            continue
        if _norm(r["invoice_number"]) != inv:
            continue
        same_gstin = _norm(bill.vendor_gstin) and _norm(bill.vendor_gstin) == _norm(r["vendor_gstin"])
        same_name = _norm(bill.vendor_name) and _norm(bill.vendor_name) == _norm(r["vendor_name"])
        if same_gstin or same_name:
            return r["id"]
    return None


# ---------------------------------------------------------------- vendor memory
def remember_vendor(conn: sqlite3.Connection, bill: Bill) -> None:
    if not bill.category:
        return
    now = datetime.now().isoformat(timespec="seconds")
    for key in vendor_keys(bill.vendor_gstin, bill.vendor_name):
        conn.execute(
            "INSERT INTO vendor_rules (key, vendor_name, category, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET category=excluded.category, "
            "vendor_name=excluded.vendor_name, updated_at=excluded.updated_at",
            (key, bill.vendor_name, bill.category, now),
        )
    conn.commit()


def lookup_category(conn: sqlite3.Connection, gstin: str, name: str) -> str:
    for key in vendor_keys(gstin, name):
        row = conn.execute("SELECT category FROM vendor_rules WHERE key=?", (key,)).fetchone()
        if row:
            return row["category"]
    return ""


def backfill_category(conn: sqlite3.Connection, bill: Bill) -> int:
    """Give existing uncategorised bills from the same vendor the category just chosen."""
    if not bill.category:
        return 0
    mine = set(vendor_keys(bill.vendor_gstin, bill.vendor_name))
    n = 0
    for r in conn.execute(
        "SELECT id, vendor_gstin, vendor_name FROM bills WHERE category IS NULL OR category=''"
    ).fetchall():
        if mine & set(vendor_keys(r["vendor_gstin"], r["vendor_name"])):
            conn.execute("UPDATE bills SET category=? WHERE id=?", (bill.category, r["id"]))
            n += 1
    conn.commit()
    return n
