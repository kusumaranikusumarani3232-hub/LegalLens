"""Tests for LegalAnalyzer."""

from pathlib import Path
from app.legal_analyzer import LegalAnalyzer
from app.models import DocumentAnalysis

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_analyzer_lease_document():
    lease_path = DATA_DIR / "residential_lease_v1.txt"
    assert lease_path.exists()
    text = lease_path.read_text(encoding="utf-8")

    analyzer = LegalAnalyzer(provider="mock")
    analysis = analyzer.analyze(text, doc_metadata={"page_count": 3, "word_count": 500})

    assert isinstance(analysis, DocumentAnalysis)
    assert "Lease" in analysis.metadata.document_type
    assert len(analysis.clauses) > 0
    assert len(analysis.obligations) > 0
    assert len(analysis.attention_points) > 0
    assert len(analysis.checklist) > 0
    assert len(analysis.lawyer_questions) > 0
    assert "LegalLens provides general information" in analysis.disclaimer


def test_missing_information_neutrality():
    # Provide a minimal document with missing jurisdiction and dates
    minimal_doc = "This is an informal note stating tenant promises to clean the room."
    analyzer = LegalAnalyzer(provider="mock")
    analysis = analyzer.analyze(minimal_doc)

    assert analysis.metadata.document_type is not None
    # Verify no alarmist language in summary or clauses
    for ap in analysis.attention_points:
        assert "dangerous trap" not in ap.observation.lower()
        assert "you will definitely lose" not in ap.why_it_matters.lower()
