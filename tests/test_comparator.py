"""Tests for DocumentComparator."""

from pathlib import Path
from app.comparator import DocumentComparator
from app.models import ComparisonResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_comparator_meaningful_changes():
    path_a = DATA_DIR / "residential_lease_v1.txt"
    path_b = DATA_DIR / "residential_lease_v2_revised.txt"
    assert path_a.exists() and path_b.exists()

    text_a = path_a.read_text(encoding="utf-8")
    text_b = path_b.read_text(encoding="utf-8")

    comparator = DocumentComparator(provider="mock")
    res = comparator.compare(text_a, text_b, doc_a_name="Lease v1", doc_b_name="Lease v2")

    assert isinstance(res, ComparisonResult)
    assert res.document_a_name == "Lease v1"
    assert res.document_b_name == "Lease v2"
    assert len(res.changes) >= 1
    
    # Check that meaningful differences were captured
    categories = [c.category.lower() for c in res.changes]
    assert any("payment" in cat or "notice" in cat or "term" in cat for cat in categories)
    
    # Check that practical implications exist
    for change in res.changes:
        assert len(change.what_this_could_mean) > 5
        assert change.source_previous
        assert change.source_new
