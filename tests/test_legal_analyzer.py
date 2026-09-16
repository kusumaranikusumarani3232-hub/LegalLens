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


def test_truncated_json_repair_regression():
    """Regression test ensuring truncated LLM JSON response is repaired and parsed into DocumentAnalysis."""
    from app.llm_provider import LLMProvider
    llm = LLMProvider(provider="mock")

    # Simulate a truncated LLM output that was cut off mid-list
    truncated_raw = """
    {
      "metadata": {
        "document_type": "Employment Agreement",
        "approximate_length": "2 pages",
        "parties_involved": "Company Inc & Employee",
        "effective_date": "2026-01-01",
        "expiration_date": "Indefinite",
        "governing_jurisdiction": "California, USA"
      },
      "summary": {
        "what_is_this": "An employment agreement.",
        "what_am_i_agreeing_to": "Perform duties.",
        "what_does_other_party_agree_to": "Pay salary.",
        "what_should_i_pay_attention_to": "Notice period."
      },
      "clauses": [
        {
          "clause_title": "Compensation",
          "category": "Payment",
          "what_the_document_says": "Salary of $100k",
          "in_simple_words": "You get paid $100k",
          "why_you_may_want_to_pay_attention": "Payment schedule",
          "source": "Section 2"
        }
      ],
      "obligations": [
        {
          "party": "Your obligations",
          "obligation": "Work 40 hours",
          "deadline_or_timeline": "Weekly",
          "source": "Section 3"
        }
      ],
      "attention_points": [
        {
          "item": "Non-compete clause",
          "observation": "Restricts competing work",
          "why_it_matters": "Limits future jobs",
          "source": "Section 5"
        }
      ],
      "checklist": [
        {
          "category": "Before Signing",
          "task": "Verify salary details",
          "detail": "Ensure bonus structure matches verbal offer"
    """

    repaired_json = llm._clean_and_parse_json(truncated_raw)
    assert isinstance(repaired_json, dict)
    assert repaired_json["metadata"]["document_type"] == "Employment Agreement"
    assert len(repaired_json["clauses"]) == 1

    analyzer = LegalAnalyzer(provider="mock")
    resilient_obj = analyzer._build_resilient_analysis(repaired_json, doc_metadata={"page_count": 2})
    assert resilient_obj.metadata.document_type == "Employment Agreement"
    assert len(resilient_obj.clauses) == 1

