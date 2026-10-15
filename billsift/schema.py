"""Shared data structures for BillSift."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

CATEGORIES = [
    "",
    "Fuel",
    "Food & Beverage",
    "Office Supplies",
    "Travel",
    "Software / Subscriptions",
    "Utilities",
    "Raw Material",
    "Rent",
    "Professional Fees",
    "Repairs & Maintenance",
    "Other",
]

# JSON schema handed to Ollama's structured-output `format` option.
BILL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor_name": {"type": "string"},
        "vendor_gstin": {"type": "string"},
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string"},
        "taxable_amount": {"type": "number"},
        "cgst": {"type": "number"},
        "sgst": {"type": "number"},
        "igst": {"type": "number"},
        "gst_rate": {"type": "number"},
        "total_amount": {"type": "number"},
    },
    "required": [
        "vendor_name",
        "vendor_gstin",
        "invoice_number",
        "invoice_date",
        "taxable_amount",
        "cgst",
        "sgst",
        "igst",
        "gst_rate",
        "total_amount",
    ],
}

EDITABLE_FIELDS = [
    "vendor_name",
    "vendor_gstin",
    "invoice_number",
    "invoice_date",
    "taxable_amount",
    "gst_rate",
    "cgst",
    "sgst",
    "igst",
    "total_amount",
    "category",
]


@dataclass
class Bill:
    source_file: str = ""
    vendor_name: str = ""
    vendor_gstin: str = ""
    invoice_number: str = ""
    invoice_date: str = ""  # ISO YYYY-MM-DD once normalised
    taxable_amount: float = 0.0
    gst_rate: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0
    total_amount: float = 0.0
    category: str = ""
    extractor: str = ""  # "llm" or "rules"
    flags: list[str] = field(default_factory=list)
    status: str = "ok"  # "ok" or "review"

    def to_dict(self) -> dict:
        return asdict(self)
