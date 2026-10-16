"""Score BillSift against the known true values of the hard sample bills.

Usage:  python scripts/eval_samples.py            (rule-based, no AI model)
        python scripts/eval_samples.py --llm gemma3:4b
        python scripts/eval_samples.py --llm qwen2.5:3b
Prints per-bill results and field accuracy, and whether the verifier flagged what it should.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from billsift.pipeline import is_problem, process_file  # noqa: E402
from billsift.store import connect  # noqa: E402

HARD = Path(__file__).resolve().parent.parent / "samples" / "hard"
FIELDS = ["vendor_gstin", "invoice_number", "invoice_date", "taxable_amount", "total_amount"]


def same(field, got, want) -> bool:
    if field in ("taxable_amount", "total_amount"):
        return abs(float(got) - float(want)) < 0.011
    return str(got).strip().upper() == str(want).strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="", help="Ollama model name; omit for rule-based only")
    args = ap.parse_args()
    manifest = json.loads((HARD / "manifest.json").read_text())
    conn = connect(":memory:")

    correct = total = 0
    flag_ok = 0
    t0 = time.time()
    for name, truth in manifest.items():
        bid, bill = process_file(conn, HARD / name, model=args.llm or "qwen2.5:3b", use_llm=bool(args.llm))
        misses = []
        for f in FIELDS:
            total += 1
            if same(f, getattr(bill, f), truth[f]):
                correct += 1
            else:
                misses.append(f"{f}: got {getattr(bill, f)!r}, want {truth[f]!r}")
        problems = [x for x in bill.flags if is_problem(x)]
        flagged = bool(problems)
        # a bill is handled correctly if (a) all fields are right and nothing is flagged,
        # or (b) it is meant to be flagged, or (c) it has a wrong field but the verifier flagged it
        safe = (not misses and not flagged and not truth["should_flag"]) or \
               (truth["should_flag"] and flagged) or (misses and flagged)
        flag_ok += safe
        status = "OK  " if not misses and not flagged else ("SAFE" if safe else "BAD ")
        print(f"[{status}] {name}  ({bill.extractor})")
        for x in misses:
            print("        miss:", x)
        for x in problems:
            print("        flag:", x)
    secs = time.time() - t0
    print(f"\nField accuracy: {correct}/{total} = {100 * correct / total:.0f}%")
    print(f"Handled safely (right, or wrong-but-flagged): {flag_ok}/{len(manifest)}")
    print(f"Time: {secs:.0f}s for {len(manifest)} bills")
    print("'SAFE' = the extraction had an error but the checker caught it. 'BAD' = a wrong value went through unflagged.")


if __name__ == "__main__":
    main()
