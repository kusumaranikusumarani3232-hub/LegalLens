"""Legal safety guardrails, disclaimer enforcement, and compliance verification."""

import re
from typing import Dict, Any, List

LEGAL_DISCLAIMER_TEXT = (
    "LegalLens provides general information and document assistance, not legal advice. "
    "AI-generated explanations may be incomplete or incorrect. "
    "For important legal decisions or disputes, consult a qualified legal professional."
)

PROHIBITED_DEFINITIVE_PATTERNS = [
    (r'\byou will (?:definitely|certainly) win\b', "you may have arguments to present"),
    (r'\byou will (?:definitely|certainly) lose\b', "there may be significant hurdles"),
    (r'\bthis clause is (?:illegal|unlawful|invalid)\b', "this clause may be subject to legal scrutiny or enforceability questions"),
    (r'\bi am your lawyer\b', "I am an AI document assistant, not an attorney"),
    (r'\bi guarantee\b', "the document suggests"),
    (r'\bdangerous trap\b', "provision worth reviewing closely"),
]


class SafetyGuardrail:
    """Enforces responsible AI, neutrality, and non-legal-advice guidelines."""

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Replaces definitive, alarmist, or unauthorized legal claims with neutral alternatives."""
        sanitized = text
        for pattern, replacement in PROHIBITED_DEFINITIVE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        return sanitized

    @classmethod
    def apply_safety_to_analysis(cls, analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all fields in an analysis dictionary adhere to safety guardrails."""
        if not isinstance(analysis_dict, dict):
            return analysis_dict

        # Sanitize summary
        if "summary" in analysis_dict and isinstance(analysis_dict["summary"], dict):
            for k, v in analysis_dict["summary"].items():
                if isinstance(v, str):
                    analysis_dict["summary"][k] = cls.sanitize_text(v)

        # Sanitize clauses
        if "clauses" in analysis_dict and isinstance(analysis_dict["clauses"], list):
            for clause in analysis_dict["clauses"]:
                if isinstance(clause, dict):
                    for k in ["what_the_document_says", "in_simple_words", "why_you_may_want_to_pay_attention"]:
                        if k in clause and isinstance(clause[k], str):
                            clause[k] = cls.sanitize_text(clause[k])

        # Sanitize attention points
        if "attention_points" in analysis_dict and isinstance(analysis_dict["attention_points"], list):
            for item in analysis_dict["attention_points"]:
                if isinstance(item, dict):
                    for k in ["observation", "why_it_matters"]:
                        if k in item and isinstance(item[k], str):
                            item[k] = cls.sanitize_text(item[k])

        # Always attach the mandatory disclaimer
        analysis_dict["disclaimer"] = LEGAL_DISCLAIMER_TEXT
        return analysis_dict

    @classmethod
    def get_disclaimer_banner(cls) -> str:
        """Returns standard markdown disclaimer banner for the UI."""
        return (
            "> ⚠️ **Legal Notice & Disclaimer**: "
            "LegalLens provides informational document understanding and comparison assistance only. "
            "It does not provide legal advice, legal opinions, or representation. "
            "Always consult a qualified legal professional before signing or making decisions regarding binding agreements."
        )
