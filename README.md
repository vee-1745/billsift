# 🧾 BillSift

**A local, privacy-first GST bill sorter for freelancers, small shops and their CAs.**

Drop in photos or PDFs of bills. BillSift reads them with open-source OCR and a small
open-weight LLM running on your own laptop, then **checks the result with plain code** before
anything reaches your spreadsheet. Your clients' financial documents never leave your machine,
and there is no per-page API bill.

## Why a verification layer?

Small local models make small mistakes: a swapped digit, a wrong field. In accounting a silently
wrong digit is worse than a missing one. So the model's output is never trusted as-is:

| Check | What it catches |
|---|---|
| GSTIN format + official mod-36 checksum | mistyped or misread GSTINs |
| Tax maths (rate x taxable = CGST+SGST / IGST) | wrong amounts, wrong rate |
| CGST = SGST, not mixed with IGST | swapped or misread tax lines |
| Taxable + tax = total (₹1 round-off allowed) | tampered or misread totals |
| **Grounding check** | every amount, GSTIN and invoice no. must literally appear in the OCR text, so a model can't invent numbers |
| Duplicate detection | the same vendor + invoice number uploaded twice |
| OCR repair | O/0, I/1, S/5 mix-ups in GSTINs and dates are fixed only when the checksum confirms it |

Anything that fails is flagged ⚠️ for a human. Everything else is exported clean.

**Vendor memory:** correct a bill's category once and every past and future bill from that vendor
gets it automatically. No fine-tuning, stored locally in SQLite.

**CA-ready export:** one Excel file with Bills, GST Summary (by financial year / quarter / month),
By Category, and a Needs Review sheet. CSV also available.

## How it works

```
photo / PDF --> RapidOCR (open-source, CPU) --> small LLM via Ollama (JSON-schema constrained)
            --> verification layer (plain code) --> SQLite --> review UI --> Excel / CSV
```

If Ollama isn't running, BillSift falls back to a rule-based extractor, so it still works
(less accurately) on clean bills with no AI model at all.

Open pieces: [RapidOCR](https://github.com/RapidAI/RapidOCR), [Ollama](https://ollama.com) with an
open-weight Qwen model, PyMuPDF, SQLite, Streamlit.

## Setup (Windows, PowerShell)

```powershell
cd C:\Users\user\billsift
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# optional but recommended: the local AI model (about 2 GB, runs on a 6 GB GPU or CPU)
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b

python scripts/make_samples.py      # generates 7 synthetic demo bills in ./samples
streamlit run app.py
```

Upload the bills from `samples/`. Three of them are deliberately broken (tampered total, mistyped
GSTIN, duplicate) to show the verifier working.

## Tests

```powershell
python -m pytest -q
```

## Project layout

```
app.py                  Streamlit UI
billsift/
  ocr.py                RapidOCR + PyMuPDF text extraction
  extract.py            Ollama LLM extraction + rule-based fallback + grounded cross-check
  verify.py             GSTIN checksum, tax maths, grounding, OCR repair
  store.py              SQLite, duplicate detection, vendor memory
  pipeline.py           file -> OCR -> extract -> verify -> database
  export.py             Excel / CSV with GST summary
scripts/make_samples.py synthetic demo bills
tests/                  pytest suite
```

## Roadmap

- Tally-importable XML export
- Marathi / Hindi bill support
- Line-item extraction
- Fine-tuned model per business

## Privacy

Everything runs locally. No bill, image or number is sent to any server. The `data/` folder
(database and uploaded bills) is git-ignored.
