"""Generate synthetic sample bills (no real customer data) so you can demo and test BillSift.

Usage:  python scripts/make_samples.py
Creates PNGs in ./samples, including deliberately broken bills the verifier should flag.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from billsift.verify import gstin_checksum_char  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples"


def gstin(prefix14: str) -> str:
    return prefix14 + gstin_checksum_char(prefix14)


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def money(x: float) -> str:
    return f"{x:,.2f}"


def make_bill(path: Path, *, vendor, vendor_gstin, inv_no, inv_date, items, rate,
              intra=True, tamper_total=None, tamper_gstin=False, messy=False):
    taxable = sum(q * p for _, q, p in items)
    tax = round(taxable * rate / 100, 2)
    total = round(taxable + tax, 2)
    shown_total = tamper_total if tamper_total is not None else total
    shown_gstin = vendor_gstin
    if tamper_gstin:  # one wrong character: the checksum must catch this
        shown_gstin = vendor_gstin[:6] + ("X" if vendor_gstin[6] != "X" else "Y") + vendor_gstin[7:]

    img = Image.new("RGB", (900, 1100), "white")
    d = ImageDraw.Draw(img)
    f_big, f, f_small = load_font(34), load_font(24), load_font(20)
    y = 40
    d.text((50, y), vendor, font=f_big, fill="black"); y += 55
    d.text((50, y), f"GSTIN: {shown_gstin}", font=f, fill="black"); y += 40
    d.text((50, y), "TAX INVOICE", font=f_big, fill="black"); y += 60
    d.text((50, y), f"Invoice No: {inv_no}", font=f, fill="black")
    d.text((520, y), f"Date: {inv_date}", font=f, fill="black"); y += 55
    d.line((50, y, 850, y), fill="black", width=2); y += 15
    d.text((50, y), "Item", font=f_small, fill="black")
    d.text((470, y), "Qty", font=f_small, fill="black")
    d.text((560, y), "Rate", font=f_small, fill="black")
    d.text((730, y), "Amount", font=f_small, fill="black"); y += 40
    for name, qty, price in items:
        d.text((50, y), name, font=f_small, fill="black")
        d.text((470, y), str(qty), font=f_small, fill="black")
        d.text((560, y), money(price), font=f_small, fill="black")
        d.text((730, y), money(qty * price), font=f_small, fill="black"); y += 38
    y += 10
    d.line((50, y, 850, y), fill="black", width=2); y += 20

    def row(label, value):
        nonlocal y
        d.text((420, y), label, font=f, fill="black")
        d.text((710, y), money(value), font=f, fill="black"); y += 40

    row("Taxable Amount", taxable)
    if intra:
        row(f"CGST @ {rate / 2:g}%", tax / 2)
        row(f"SGST @ {rate / 2:g}%", tax / 2)
    else:
        row(f"IGST @ {rate:g}%", tax)
    y += 10
    d.text((420, y), "Grand Total", font=f_big, fill="black")
    d.text((680, y), money(shown_total), font=f_big, fill="black")
    y += 90
    d.text((50, y), "Thank you for your business!", font=f_small, fill="black")

    if messy:  # slight rotation + blur + noise, like a phone photo
        img = img.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor="white")
        img = img.filter(ImageFilter.GaussianBlur(0.8))
    img.save(path)


def main():
    random.seed(7)
    OUT.mkdir(exist_ok=True)
    pump = gstin("27AABCP1234A1Z")
    cafe = gstin("27AAECC5678B1Z")
    stat = gstin("29AAGCS4321D1Z")
    soft = gstin("27AAHCT9988K1Z")

    make_bill(OUT / "01_petrol_pump_ok.png", vendor="Shree Balaji Fuels", vendor_gstin=pump,
              inv_no="BF/2026/0412", inv_date="03/10/2026",
              items=[("Petrol (Premium)", 20, 105.50)], rate=18)
    make_bill(OUT / "02_cafe_ok.png", vendor="Cafe Chaicoffee", vendor_gstin=cafe,
              inv_no="CC-7781", inv_date="05-10-2026",
              items=[("Cold Coffee", 2, 180.00), ("Veg Sandwich", 2, 150.00)], rate=5, messy=True)
    make_bill(OUT / "03_stationery_igst_ok.png", vendor="Sharma Stationers", vendor_gstin=stat,
              inv_no="SS/118", inv_date="07 Oct 2026",
              items=[("A4 Paper Ream", 10, 285.00), ("Gel Pens (box)", 5, 120.00)],
              rate=12, intra=False)
    make_bill(OUT / "04_software_ok.png", vendor="TechNova Solutions", vendor_gstin=soft,
              inv_no="TN-2026-0099", inv_date="09/10/2026",
              items=[("Annual hosting plan", 1, 12000.00)], rate=18, messy=True)
    # --- bills the verifier should catch ---
    make_bill(OUT / "05_BAD_total_tampered.png", vendor="Sharma Stationers", vendor_gstin=stat,
              inv_no="SS/119", inv_date="08/10/2026",
              items=[("Printer Cartridge", 2, 1850.00)], rate=12, tamper_total=4600.00)
    make_bill(OUT / "06_BAD_gstin_typo.png", vendor="Shree Balaji Fuels", vendor_gstin=pump,
              inv_no="BF/2026/0413", inv_date="10/10/2026",
              items=[("Diesel", 30, 92.40)], rate=18, tamper_gstin=True)
    # exact duplicate of bill 01 (same vendor + invoice number): duplicate detector should fire
    make_bill(OUT / "07_duplicate_of_01.png", vendor="Shree Balaji Fuels", vendor_gstin=pump,
              inv_no="BF/2026/0412", inv_date="03/10/2026",
              items=[("Petrol (Premium)", 20, 105.50)], rate=18, messy=True)
    print(f"Wrote {len(list(OUT.glob('*.png')))} sample bills to {OUT}")


if __name__ == "__main__":
    main()
