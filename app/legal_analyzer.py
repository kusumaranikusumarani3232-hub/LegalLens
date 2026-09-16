"""Legal Document Analyzer for comprehensive structured document breakdown."""

from typing import Dict, Any, Optional
from app.models import DocumentAnalysis, DocumentMetadata, PlainEnglishSummary, ClauseItem, ObligationItem, AttentionPoint, ActionChecklistItem, LawyerQuestion
from app.llm_provider import LLMProvider, LLMError
from app.prompts import DOCUMENT_ANALYSIS_PROMPT
from app.safety import SafetyGuardrail
from app.config import config


class LegalAnalyzer:
    """Orchestrates comprehensive single-document analysis."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.llm = LLMProvider(provider=provider, api_key=api_key)

    def analyze(self, full_text: str, doc_metadata: Optional[Dict[str, Any]] = None) -> DocumentAnalysis:
        """
        Takes extracted text and returns a validated DocumentAnalysis object.
        """
        if not full_text or not full_text.strip():
            raise ValueError("Document text is empty.")

        # Truncate if exceptionally large to protect context limits
        truncated_text = full_text
        if len(truncated_text) > config.MAX_DOC_CHARS_ANALYSIS:
            truncated_text = truncated_text[:config.MAX_DOC_CHARS_ANALYSIS] + "\n\n... [Remaining document truncated for length] ..."

        prompt = DOCUMENT_ANALYSIS_PROMPT.format(document_text=truncated_text)
        raw_json = self.llm.generate_json(prompt)

        # Apply safety filters
        sanitized_json = SafetyGuardrail.apply_safety_to_analysis(raw_json)

        # If doc_metadata was provided from processor, ensure length / page stats are accurate
        if doc_metadata and "metadata" in sanitized_json:
            meta = sanitized_json["metadata"]
            if not meta.get("approximate_length") or meta.get("approximate_length") == "Not found in the document.":
                meta["approximate_length"] = f"{doc_metadata.get('page_count', 1)} pages (~{doc_metadata.get('word_count', 0):,} words)"

        # Parse into Pydantic models with graceful fallbacks
        try:
            return DocumentAnalysis(**sanitized_json)
        except Exception as parse_err:
            # Construct resilient fallback model
            return self._build_resilient_analysis(sanitized_json, doc_metadata)

    def _build_resilient_analysis(self, data: Dict[str, Any], doc_metadata: Optional[Dict[str, Any]]) -> DocumentAnalysis:
        """Constructs DocumentAnalysis handling partially missing fields."""
        m_data = data.get("metadata", {})
        metadata = DocumentMetadata(
            document_type=m_data.get("document_type", "Legal Document"),
            approximate_length=m_data.get("approximate_length", f"{doc_metadata.get('page_count', 1) if doc_metadata else 1} pages"),
            parties_involved=m_data.get("parties_involved", "Not found in the document."),
            effective_date=m_data.get("effective_date", "Not found in the document."),
            expiration_date=m_data.get("expiration_date", "Not found in the document."),
            governing_jurisdiction=m_data.get("governing_jurisdiction", "Not found in the document.")
        )

        s_data = data.get("summary", {})
        summary = PlainEnglishSummary(
            what_is_this=s_data.get("what_is_this", "A legal agreement outlining terms and conditions between parties."),
            what_am_i_agreeing_to=s_data.get("what_am_i_agreeing_to", "Review the document clauses carefully for your specific commitments."),
            what_does_other_party_agree_to=s_data.get("what_does_other_party_agree_to", "The other party agrees to perform the designated deliverables or obligations."),
            what_should_i_pay_attention_to=s_data.get("what_should_i_pay_attention_to", "Pay attention to notice periods, renewal conditions, and liability limits.")
        )

        clauses = []
        for c in data.get("clauses", []):
            if isinstance(c, dict):
                clauses.append(ClauseItem(
                    clause_title=c.get("clause_title", "Key Provision"),
                    category=c.get("category", "Responsibilities"),
                    what_the_document_says=c.get("what_the_document_says", "Stated in agreement text."),
                    in_simple_words=c.get("in_simple_words", "Establishes terms between parties."),
                    why_you_may_want_to_pay_attention=c.get("why_you_may_want_to_pay_attention", "Affects operational or legal rights."),
                    source=c.get("source", "Document body")
                ))

        obligations = []
        for o in data.get("obligations", []):
            if isinstance(o, dict):
                obligations.append(ObligationItem(
                    party=o.get("party", "Your obligations"),
                    obligation=o.get("obligation", "Comply with agreement terms."),
                    deadline_or_timeline=o.get("deadline_or_timeline", "Not specified in the document."),
                    source=o.get("source", "Document body")
                ))

        attention_points = []
        for a in data.get("attention_points", []):
            if isinstance(a, dict):
                attention_points.append(AttentionPoint(
                    item=a.get("item", "Provisional Term"),
                    observation=a.get("observation", "Review stated clauses."),
                    why_it_matters=a.get("why_it_matters", "Has contractual implications."),
                    source=a.get("source", "Document body")
                ))

        checklist = []
        for ch in data.get("checklist", []):
            if isinstance(ch, dict):
                checklist.append(ActionChecklistItem(
                    category=ch.get("category", "Before Signing"),
                    task=ch.get("task", "Review terms with counterparty"),
                    detail=ch.get("detail", "Ensure mutual understanding of commitments."),
                    completed=False
                ))

        lawyer_questions = []
        for q in data.get("lawyer_questions", []):
            if isinstance(q, dict):
                lawyer_questions.append(LawyerQuestion(
                    category=q.get("category", "General Review"),
                    question=q.get("question", "What are the primary risks associated with this contract?"),
                    context=q.get("context", "Reviewing general provisions before execution.")
                ))

        return DocumentAnalysis(
            metadata=metadata,
            summary=summary,
            clauses=clauses,
            obligations=obligations,
            attention_points=attention_points,
            checklist=checklist,
            lawyer_questions=lawyer_questions
        )
