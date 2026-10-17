"""Pre-processed sample bills for the public demo.

Free hosts have tiny CPUs and little RAM (Render's free tier: 0.1 CPU, 512 MB), so OCR-ing the sample
bills live would be slow and could run out of memory. Instead the samples are processed once with the
real pipeline (rule-based extractor + verifier) and stored in samples/hard/precomputed.json; the demo
loads that file instantly. Regenerate it after changing the pipeline:

    python -m billsift.demo
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .ocr import IMAGE_EXTS
from .pipeline import process_file
from .schema import Bill
from .store import BILL_COLUMNS, connect, insert_bill, list_bills

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples" / "hard"
CACHE = SAMPLES_DIR / "precomputed.json"
SUPPORTED = IMAGE_EXTS | {".pdf"}


def sample_files(samples_dir: Path = SAMPLES_DIR) -> list[Path]:
    if not samples_dir.exists():
        return []
    return sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in SUPPORTED)


def build(samples_dir: Path = SAMPLES_DIR, out: Path = CACHE) -> int:
    """Run the real pipeline (no AI model) over the samples and save the results as JSON."""
    conn = connect(":memory:")
    for f in sample_files(samples_dir):
        process_file(conn, f, use_llm=False)
    rows = []
    for r in reversed(list_bills(conn)):  # list_bills is newest first; keep processing order
        d = {c: r[c] for c in BILL_COLUMNS}
        d.update(flags=r["flags"], ocr_text=r["ocr_text"], file=Path(r["stored_path"]).name)
        rows.append(d)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def load(conn: sqlite3.Connection, samples_dir: Path = SAMPLES_DIR, cache: Path = CACHE) -> int:
    """Insert the pre-processed sample bills into a session's database. Instant."""
    rows = json.loads(cache.read_text(encoding="utf-8"))
    for d in rows:
        bill = Bill(**{c: d[c] for c in BILL_COLUMNS}, flags=list(d["flags"]))
        insert_bill(conn, bill, ocr_text=d["ocr_text"], stored_path=str(samples_dir / d["file"]))
    return len(rows)


if __name__ == "__main__":
    print(f"Wrote {build()} pre-processed sample bills to {CACHE}")
