"""Tests for SafetyGuardrail and compliance."""

from app.safety import SafetyGuardrail, LEGAL_DISCLAIMER_TEXT


def test_sanitize_definitive_legal_claims():
    toxic_input = "You will definitely win if you go to court because this clause is illegal and a dangerous trap!"
    sanitized = SafetyGuardrail.sanitize_text(toxic_input)

    assert "you will definitely win" not in sanitized.lower()
    assert "illegal" not in sanitized.lower()
    assert "dangerous trap" not in sanitized.lower()
    assert "you may have arguments" in sanitized.lower()


def test_mandatory_disclaimer_attachment():
    raw_dict = {
        "metadata": {},
        "summary": {"what_is_this": "A contract."},
        "clauses": [{"what_the_document_says": "You will definitely lose."}]
    }
    safe_dict = SafetyGuardrail.apply_safety_to_analysis(raw_dict)

    assert safe_dict["disclaimer"] == LEGAL_DISCLAIMER_TEXT
    assert "you will definitely lose" not in safe_dict["clauses"][0]["what_the_document_says"].lower()


def test_disclaimer_banner_presence():
    banner = SafetyGuardrail.get_disclaimer_banner()
    assert "Legal Notice & Disclaimer" in banner
    assert "informational document understanding" in banner
