"""OCR with RapidOCR (open-source, runs on CPU via ONNX) and PyMuPDF for PDFs."""
from __future__ import annotations

import io
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


def ocr_pdf(path: str | Path) -> str:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            text = page.get_text().strip()
            if len(text) >= 40:  # digital PDF: use the embedded text, no OCR needed
                pages.append(text)
            else:  # scanned PDF: render and OCR
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                pages.append(ocr_array(_to_array(img)))
    return "\n\n".join(pages)


def read_document(path: str | Path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return ocr_pdf(p)
    if ext in IMAGE_EXTS:
        return ocr_image(p)
    raise ValueError(f"Unsupported file type: {ext}")
