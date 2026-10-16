"""OCR with RapidOCR (open-source, runs on CPU via ONNX) and PyMuPDF for PDFs."""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
from PIL import Image

_engine = None
MAX_SIDE = 2200  # downscale huge phone photos so OCR stays fast
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def _to_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return np.array(img)


def _lines_from_result(result) -> str:
    """Rebuild reading-order lines so 'CGST 9%   90.00' stays on one line."""
    items = []
    for box, text, _score in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        height = max(ys) - min(ys)
        items.append(((min(ys) + max(ys)) / 2, min(xs), max(height, 1.0), text))
    items.sort(key=lambda t: t[0])

    lines: list[list[tuple[float, str]]] = []
    line_y = None
    line_h = 1.0
    for yc, x, h, text in items:
        if line_y is not None and abs(yc - line_y) <= 0.6 * max(h, line_h):
            lines[-1].append((x, text))
            line_y = (line_y + yc) / 2
        else:
            lines.append([(x, text)])
            line_y, line_h = yc, h
    return "\n".join("  ".join(t for _, t in sorted(line)) for line in lines)


def ocr_array(arr: np.ndarray) -> str:
    result, _ = _get_engine()(arr)
    if not result:
        return ""
    return _lines_from_result(result)


def ocr_image(path: str | Path) -> str:
    with Image.open(path) as img:
        return ocr_array(_to_array(img))


def _pdf_lines(page) -> str:
    """Embedded PDF text rebuilt into visual lines (label and value in separate table cells
    would otherwise come out on separate lines and break 'Taxable Value   1,000.00')."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, wordno)
    items = sorted(((w[1] + w[3]) / 2, w[0], w[2], w[3] - w[1], w[4]) for w in words)
    lines: list[list[tuple[float, float, str]]] = []
    cy, ch = None, 1.0
    for yc, x0, x1, h, t in items:
        if cy is not None and abs(yc - cy) <= 0.5 * max(h, ch):
            lines[-1].append((x0, x1, t))
            cy = (cy + yc) / 2
        else:
            lines.append([(x0, x1, t)])
            cy, ch = yc, h
    out = []
    for line in lines:
        line.sort()
        s, prev = line[0][2], line[0][1]
        for x0, x1, t in line[1:]:
            s += ("  " if x0 - prev > 10 else " ") + t
            prev = x1
        out.append(s)
    return "\n".join(out)


def ocr_pdf(path: str | Path) -> str:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            text = _pdf_lines(page).strip()
            if len(text) >= 40:  # digital PDF: use the embedded text, no OCR needed
                pages.append(text)
            else:  # scanned PDF: render and OCR
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                pages.append(ocr_array(_to_array(img)))
    return "\n\n".join(pages)


_DOT_THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3})((?:\.\d{3})+)(\.\d{2})(?!\d)")


def fix_amount_separators(text: str) -> str:
    """OCR often prints the thousands comma as a dot: '3.339.40' -> '3,339.40'."""
    return _DOT_THOUSANDS.sub(lambda m: m.group(1) + m.group(2).replace(".", ",") + m.group(3), text)


def read_document(path: str | Path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return fix_amount_separators(ocr_pdf(p))
    if ext in IMAGE_EXTS:
        return fix_amount_separators(ocr_image(p))
    raise ValueError(f"Unsupported file type: {ext}")
