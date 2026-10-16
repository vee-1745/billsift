"""Generate a HARDER batch of synthetic bills: different layouts, phone-photo damage, PDFs.

Usage:  python scripts/make_hard_samples.py
Writes samples/hard/*  plus samples/hard/manifest.json (the true values, for scripts/eval_samples.py).
Everything is fake: invented vendors, invented GSTINs with valid checksums.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from billsift.verify import gstin_checksum_char  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples" / "hard"
MANIFEST: dict[str, dict] = {}


def gstin(p14: str) -> str:
    return p14 + gstin_checksum_char(p14)


def font(size: int, mono=False, bold=False):
    names = (["DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"] if mono
             else ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"])
    for n in names + ["arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def m(x: float, prefix: str = "") -> str:
    return f"{prefix}{x:,.2f}"


def record(name, *, vendor_gstin, inv_no, iso_date, taxable, total, flagged=False, note=""):
    MANIFEST[name] = dict(vendor_gstin=vendor_gstin, invoice_number=inv_no, invoice_date=iso_date,
                          taxable_amount=taxable, total_amount=total, should_flag=flagged, note=note)


# ------------------------------------------------------------------ photo damage
def phone_photo(img: Image.Image, *, rot=3.0, skew=0.06, shadow=0.35, noise=9, blur=1.0,
                quality=45, contrast=1.0) -> Image.Image:
    import cv2

    rgb = np.array(img.convert("RGB")).astype(np.float32)
    h, w = rgb.shape[:2]
    if contrast != 1.0:
        rgb = 255 - (255 - rgb) * contrast
    # soft shadow gradient across the page (hand / phone shadow)
    grad = np.linspace(1.0, 1.0 - shadow, w)[None, :, None]
    rgb = rgb * grad
    rgb += np.random.normal(0, noise, rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # perspective: top edge slightly narrower than bottom, like a photo taken from an angle
    pad = 60
    canvas = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(170, 160, 150))
    H, W = canvas.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    d = skew * W
    dst = np.float32([[d, d * 0.4], [W - d * 0.3, 0], [W, H], [0, H - d * 0.5]])
    canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst), (W, H),
                                 borderValue=(170, 160, 150))
    pil = Image.fromarray(canvas).rotate(rot, expand=True, fillcolor=(170, 160, 150))
    if blur:
        pil = pil.filter(ImageFilter.GaussianBlur(blur))
    return pil, quality


def save_jpeg(img_q, path: Path):
    img, q = img_q
    img.save(path, "JPEG", quality=q)


# ------------------------------------------------------------------ renderers
def thermal(path: Path, *, name, gst, inv_no, date_txt, items, rate, round_off, damage=None):
    """Narrow monospace till receipt, ALL CAPS, 'NET AMOUNT' and ROUND OFF."""
    taxable = round(sum(q * p for _, q, p in items), 2)
    half = round(taxable * rate / 200, 2)
    raw = taxable + 2 * half
    total = round(raw + round_off, 2)
    img = Image.new("RGB", (520, 900), "white")
    d = ImageDraw.Draw(img)
    f, fb = font(21, mono=True), font(24, mono=True, bold=True)
    y = 25
    for txt, ft in [(name.upper(), fb), ("GSTIN:" + gst, f), ("-" * 32, f),
                    (f"BILL NO: {inv_no}", f), (f"DT: {date_txt}", f), ("-" * 32, f),
                    ("ITEM         QTY   AMT", f)]:
        d.text((20, y), txt, font=ft, fill="black"); y += 34
    for n, q, p in items:
        d.text((20, y), f"{n.upper()[:12]:<12}  {q:>3}  {q * p:>8,.2f}", font=f, fill="black"); y += 32
    d.text((20, y), "-" * 32, font=f, fill="black"); y += 34
    for lab, val in [("SUB TOTAL", taxable), (f"CGST {rate / 2:g}%", half), (f"SGST {rate / 2:g}%", half),
                     ("ROUND OFF", round_off)]:
        d.text((20, y), f"{lab:<18}{val:>12,.2f}", font=f, fill="black"); y += 32
    d.text((20, y), "=" * 32, font=f, fill="black"); y += 34
    d.text((20, y), f"{'NET AMOUNT':<14}{total:>10,.2f}", font=fb, fill="black"); y += 60
    d.text((20, y), "THANK YOU VISIT AGAIN", font=f, fill="black")
    out = phone_photo(img, **damage) if damage else (img, 95)
    save_jpeg(out, path)
    return taxable, total


def invoice(path: Path, *, name, gst, buyer=None, inv_label="Invoice No", inv_no, date_txt,
            items, discount_pct=0, rate, igst=False, money_prefix="", total_label="Grand Total",
            hsn=False, damage=None, tamper_tax=None, pdf=None):
    """Classic A4-style tax invoice with options."""
    gross = round(sum(q * p for _, q, p in items), 2)
    disc = round(gross * discount_pct / 100, 2)
    taxable = round(gross - disc, 2)
    tax = round(taxable * rate / 100, 2)
    shown_tax = tamper_tax if tamper_tax is not None else tax
    total = round(taxable + shown_tax, 2)

    img = Image.new("RGB", (900, 1200), "white")
    d = ImageDraw.Draw(img)
    f_big, f, f_s = font(34, bold=True), font(24), font(20)
    y = 40
    d.text((50, y), name, font=f_big, fill="black"); y += 52
    d.text((50, y), "12, Market Road, Nagpur 440001", font=f_s, fill="black"); y += 32
    d.text((50, y), f"GSTIN: {gst}", font=f, fill="black"); y += 46
    d.text((50, y), "TAX INVOICE", font=f_big, fill="black"); y += 56
    d.text((50, y), f"{inv_label}: {inv_no}", font=f, fill="black")
    d.text((540, y), f"Date: {date_txt}", font=f, fill="black"); y += 48
    if buyer:
        d.text((50, y), "Bill To:", font=f_s, fill="black"); y += 28
        d.text((50, y), buyer[0], font=f, fill="black"); y += 32
        d.text((50, y), f"Buyer GSTIN: {buyer[1]}", font=f_s, fill="black"); y += 40
    d.line((50, y, 850, y), fill="black", width=2); y += 12
    cols = [("Description", 50), ("HSN", 380), ("Qty", 480), ("Rate", 570), ("Amount", 730)] if hsn \
        else [("Description", 50), ("Qty", 470), ("Rate", 560), ("Amount", 730)]
    for t, x in cols:
        d.text((x, y), t, font=f_s, fill="black")
    y += 36
    for n, q, p in items:
        d.text((50, y), n, font=f_s, fill="black")
        if hsn:
            d.text((380, y), str(random.choice([8471, 8473, 4820, 3926])), font=f_s, fill="black")
            d.text((480, y), str(q), font=f_s, fill="black")
            d.text((570, y), m(p), font=f_s, fill="black")
        else:
            d.text((470, y), str(q), font=f_s, fill="black")
            d.text((560, y), m(p), font=f_s, fill="black")
        d.text((730, y), m(q * p, money_prefix), font=f_s, fill="black"); y += 36
    y += 8
    d.line((50, y, 850, y), fill="black", width=2); y += 18

    def row(lab, val, big=False):
        nonlocal y
        ft = f_big if big else f
        d.text((400, y), lab, font=ft, fill="black")
        d.text((690 if not big else 650, y), m(val, money_prefix), font=ft, fill="black"); y += 44 if big else 38

    row("Gross Amount", gross)
    if disc:
        row(f"Discount {discount_pct}%", disc)
    row("Taxable Value", taxable)
    if igst:
        row(f"IGST {rate:g}%", shown_tax)
    else:
        row(f"CGST {rate / 2:g}%", shown_tax / 2)
        row(f"SGST {rate / 2:g}%", shown_tax / 2)
    y += 8
    row(total_label, total, big=True)
    y += 30
    d.text((50, y), "Subject to Nagpur jurisdiction. E&OE.", font=f_s, fill="black")

    if pdf == "scanned":
        img.save(path.with_suffix(".png")); _png_to_pdf(path.with_suffix(".png"), path); path.with_suffix(".png").unlink()
    elif damage:
        save_jpeg(phone_photo(img, **damage), path)
    else:
        img.save(path)
    return taxable, total, shown_tax != tax


def _png_to_pdf(png: Path, pdf: Path):
    import pymupdf

    doc = pymupdf.open()
    im = Image.open(png)
    page = doc.new_page(width=im.width * 0.75, height=im.height * 0.75)
    page.insert_image(page.rect, filename=str(png))
    doc.save(pdf)
    doc.close()


def digital_pdf(path: Path, *, name, gst, inv_no, date_txt, taxable, rate):
    """A real text PDF (like an Amazon/Flipkart invoice): no OCR needed."""
    import pymupdf

    tax = round(taxable * rate / 100, 2)
    total = round(taxable + tax, 2)
    lines = [name, "Sold By: " + name, f"GSTIN: {gst}", "TAX INVOICE / BILL OF SUPPLY", "",
             f"Invoice Number: {inv_no}", f"Invoice Date: {date_txt}", "",
             "Description            Qty     Net Amount     Tax Rate     Tax Amount     Total",
             f"Wireless Mouse          1      {taxable:,.2f}       {rate:g}%        {tax:,.2f}      {total:,.2f}",
             "", f"Taxable Amount  {taxable:,.2f}", f"IGST {rate:g}%  {tax:,.2f}",
             f"Total Amount  {total:,.2f}", "", "Amount in words: Rupees only"]
    doc = pymupdf.open()
    page = doc.new_page()
    y = 70
    for ln in lines:
        page.insert_text((50, y), ln, fontsize=11)
        y += 18
    doc.save(path)
    doc.close()
    return taxable, total


# ------------------------------------------------------------------ build the batch
def main():
    random.seed(11)
    np.random.seed(11)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()

    g = {
        "hotel": gstin("27AAFCS7788P1Z"), "mobile": gstin("27AABCM2211Q1Z"),
        "print": gstin("27AAGCP4455R1Z"), "elec": gstin("29AAHCE9090L1Z"),
        "med": gstin("27AAICM1357N1Z"), "buyer": gstin("27AAJCV8642K1Z"),
        "book": gstin("27AAKCB3030M1Z"), "tools": gstin("24AALCT5151H1Z"),
        "cloud": gstin("27AAMCC6060J1Z"), "garage": gstin("27AANCG7272D1Z"),
        "dist": gstin("27AAOCD8484B1Z"),
    }

    n = "H01_thermal_receipt.jpg"
    t, tot = thermal(OUT / n, name="Hotel Sai Prasad", gst=g["hotel"], inv_no="4471", date_txt="12/10/26",
                     items=[("Masala Dosa", 2, 90.0), ("Filter Coffee", 2, 40.0), ("Paneer Thali", 1, 220.0)],
                     rate=5, round_off=-0.20, damage=dict(rot=1.5, skew=0.02, shadow=0.15, noise=6, blur=0.7, quality=60))
    record(n, vendor_gstin=g["hotel"], inv_no="4471", iso_date="2026-10-12", taxable=t, total=tot,
           note="thermal receipt, caps, round-off line")

    n = "H02_buyer_and_seller_gstin.png"
    t, tot, _ = invoice(OUT / n, name="Mobile Mart Nagpur", gst=g["mobile"], buyer=("Vedant Enterprises", g["buyer"]),
                        inv_label="Invoice #", inv_no="MM-25-1182", date_txt="11-10-2026",
                        items=[("Phone Case", 2, 349.0), ("Tempered Glass", 2, 199.0), ("Charger 25W", 1, 1299.0)],
                        rate=18)
    record(n, vendor_gstin=g["mobile"], inv_no="MM-25-1182", iso_date="2026-10-11", taxable=t, total=tot,
           note="buyer GSTIN printed too; the seller is the vendor")

    n = "H03_rs_prefix_amounts.png"
    t, tot, _ = invoice(OUT / n, name="Prime Print & Copy", gst=g["print"], inv_no="PPC/0311", date_txt="09/10/2026",
                        items=[("Colour Printouts A4", 120, 8.0), ("Spiral Binding", 3, 60.0)], rate=12,
                        money_prefix="Rs. ", total_label="Amount Payable")
    record(n, vendor_gstin=g["print"], inv_no="PPC/0311", iso_date="2026-10-09", taxable=t, total=tot,
           note="amounts printed with 'Rs.' prefix, label 'Amount Payable'")

    n = "H04_discount_igst.png"
    t, tot, _ = invoice(OUT / n, name="Bright Electricals", gst=g["elec"], inv_no="BE/0455", date_txt="10/10/2026",
                        items=[("LED Panel 18W", 10, 420.0), ("Ceiling Fan", 2, 2150.0), ("Switch Board", 5, 90.0)],
                        discount_pct=10, rate=18, igst=True)
    record(n, vendor_gstin=g["elec"], inv_no="BE/0455", iso_date="2026-10-10", taxable=t, total=tot,
           note="10% discount line, inter-state IGST")

    n = "H05_month_name_date.png"
    t, tot, _ = invoice(OUT / n, name="City Medicos", gst=g["med"], inv_label="Inv. No.", inv_no="AB/25-26/0098",
                        date_txt="12-Oct-26", items=[("Paracetamol Strips", 10, 28.0), ("ORS Sachets", 20, 22.0)],
                        rate=12, total_label="Total Amount")
    record(n, vendor_gstin=g["med"], inv_no="AB/25-26/0098", iso_date="2026-10-12", taxable=t, total=tot,
           note="dd-Mon-yy date, 'Inv. No.' label")

    n = "H06_hsn_column_slash_date.png"
    t, tot, _ = invoice(OUT / n, name="Orange Book House", gst=g["book"], inv_no="OBH-7781", date_txt="2026/10/11",
                        items=[("Data Structures Textbook", 2, 650.0), ("Notebook A4", 12, 55.0)],
                        rate=5, hsn=True, money_prefix="")
    record(n, vendor_gstin=g["book"], inv_no="OBH-7781", iso_date="2026-10-11", taxable=t, total=tot,
           note="HSN column adds many extra numbers, yyyy/mm/dd date")

    n = "H07_phone_photo_skewed_shadow.jpg"
    t, tot, _ = invoice(OUT / n, name="Gupta Tools & Hardware", gst=g["tools"], inv_no="GT/2210", date_txt="08/10/2026",
                        items=[("Drill Bit Set", 1, 1450.0), ("Hammer 500g", 2, 320.0), ("Safety Gloves", 4, 95.0)],
                        rate=18, damage=dict(rot=4.0, skew=0.07, shadow=0.45, noise=11, blur=1.2, quality=40))
    record(n, vendor_gstin=g["tools"], inv_no="GT/2210", iso_date="2026-10-08", taxable=t, total=tot,
           note="phone photo: perspective skew, heavy shadow, noise, JPEG q40")

    n = "H08_faded_low_contrast.jpg"
    t, tot = thermal(OUT / n, name="Sunrise Bakery", gst=g["dist"], inv_no="S-0917", date_txt="10/10/26",
                     items=[("Veg Puff", 6, 25.0), ("Cake Slice", 4, 60.0), ("Bread", 2, 45.0)],
                     rate=5, round_off=0.00, damage=dict(rot=-3.5, skew=0.04, shadow=0.3, noise=10, blur=1.4,
                                                          quality=35, contrast=0.55))
    record(n, vendor_gstin=g["dist"], inv_no="S-0917", iso_date="2026-10-10", taxable=t, total=tot,
           note="faded thermal print, low contrast, rotated")

    n = "H09_digital_pdf.pdf"
    t, tot = digital_pdf(OUT / n, name="CloudNine Hosting", gst=g["cloud"], inv_no="CN-2026-5521",
                         date_txt="07/10/2026", taxable=8999.0, rate=18)
    record(n, vendor_gstin=g["cloud"], inv_no="CN-2026-5521", iso_date="2026-10-07", taxable=t, total=tot,
           note="digital text PDF (embedded text, no OCR needed)")

    n = "H10_scanned_pdf.pdf"
    t, tot, _ = invoice(OUT / n, name="Auto Care Garage", gst=g["garage"], inv_no="ACG/0066", date_txt="06/10/2026",
                        items=[("Engine Oil 3.5L", 1, 1850.0), ("Oil Filter", 1, 380.0), ("Labour", 1, 600.0)],
                        rate=18, pdf="scanned")
    record(n, vendor_gstin=g["garage"], inv_no="ACG/0066", iso_date="2026-10-06", taxable=t, total=tot,
           note="image-only scanned PDF, needs OCR")

    n = "H11_BAD_tax_wrong.png"
    t, tot, _ = invoice(OUT / n, name="Prime Print & Copy", gst=g["print"], inv_no="PPC/0312", date_txt="10/10/2026",
                        items=[("Flex Banner", 2, 1200.0)], rate=18, tamper_tax=300.00)
    record(n, vendor_gstin=g["print"], inv_no="PPC/0312", iso_date="2026-10-10", taxable=t, total=tot,
           flagged=True, note="tax printed as 300 instead of 432: must be flagged")

    (OUT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
    print(f"Wrote {len(MANIFEST)} hard sample bills to {OUT}")


if __name__ == "__main__":
    main()
