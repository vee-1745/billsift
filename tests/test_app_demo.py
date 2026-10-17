"""The public demo mode must give every visitor a private, empty workspace."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def metrics(at):
    return {m.label: m.value for m in at.metric}


@pytest.mark.skipif(not (ROOT / "samples" / "hard").exists(), reason="hard samples missing")
def test_demo_mode_is_private_per_visitor(monkeypatch):
    monkeypatch.setenv("BILLSIFT_DEMO", "1")
    one = AppTest.from_file(APP, default_timeout=180).run()
    assert not one.exception
    assert any("Public demo" in w.value for w in one.sidebar.warning)

    next(b for b in one.button if "sample bills" in b.label).click()
    one.run()
    assert not one.exception
    assert metrics(one)["Bills"] == "11"

    # a second visitor must start with nothing: no shared database
    two = AppTest.from_file(APP, default_timeout=180).run()
    assert not two.exception
    assert not two.metric
    assert any("sample bills" in b.label for b in two.button)


@pytest.mark.skipif(not (ROOT / "samples" / "hard" / "precomputed.json").exists(), reason="no precomputed demo data")
def test_precomputed_demo_data_is_fresh(tmp_path):
    """samples/hard/precomputed.json must match what the pipeline produces today.
    If this fails, run:  python -m billsift.demo   and commit the result."""
    import json

    from billsift import demo

    fresh = tmp_path / "fresh.json"
    demo.build(out=fresh)
    keys = ["file", "vendor_gstin", "invoice_number", "invoice_date", "taxable_amount", "total_amount", "status", "flags"]
    pick = lambda path: [{k: r[k] for k in keys} for r in json.loads(path.read_text(encoding="utf-8"))]  # noqa: E731
    assert pick(fresh) == pick(demo.CACHE)


def test_uploads_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("BILLSIFT_DEMO", "1")
    monkeypatch.setenv("BILLSIFT_DEMO_UPLOADS", "0")
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("Uploads are switched off" in i.value for i in at.info)
