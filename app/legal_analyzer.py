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
        
        try:
            raw_json = self.llm.generate_json(prompt)
        except Exception as err:
            # If the LLM call or JSON parsing fails completely, raise LLMError with clear message
            raise LLMError(f"Legal document analysis failed: {str(err)}") from err

        # Apply safety filters
        sanitized_json = SafetyGuardrail.apply_safety_to_analysis(raw_json)

        # If doc_metadata was provided from processor, ensure length / page stats are accurate
        if doc_metadata and "metadata" in sanitized_json:
            meta = sanitized_json["metadata"]
            if isinstance(meta, dict) and (not meta.get("approximate_length") or meta.get("approximate_length") == "Not found in the document."):
                meta["approximate_length"] = f"{doc_metadata.get('page_count', 1)} pages (~{doc_metadata.get('word_count', 0):,} words)"

        # Parse into Pydantic models with graceful fallbacks
        try:
            return DocumentAnalysis(**sanitized_json)
        except Exception:
            # Construct resilient fallback model using whatever fields were returned
            return self._build_resilient_analysis(sanitized_json, doc_metadata)

    def _build_resilient_analysis(self, data: Dict[str, Any], doc_metadata: Optional[Dict[str, Any]]) -> DocumentAnalysis:
        """Constructs DocumentAnalysis preserving all recovered fields while handling partial keys."""
        if not isinstance(data, dict):
            data = {}

        m_data = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        metadata = DocumentMetadata(
            document_type=m_data.get("document_type") or "Legal Document",
            approximate_length=m_data.get("approximate_length") or (f"{doc_metadata.get('page_count', 1) if doc_metadata else 1} pages"),
            parties_involved=m_data.get("parties_involved") or "Not found in the document.",
            effective_date=m_data.get("effective_date") or "Not found in the document.",
            expiration_date=m_data.get("expiration_date") or "Not found in the document.",
            governing_jurisdiction=m_data.get("governing_jurisdiction") or "Not found in the document."
        )

        s_data = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        summary = PlainEnglishSummary(
            what_is_this=s_data.get("what_is_this") or s_data.get("overview") or s_data.get("purpose") or "A legal agreement outlining terms and conditions between parties.",
            what_am_i_agreeing_to=s_data.get("what_am_i_agreeing_to") or s_data.get("user_obligations") or "Review the document clauses carefully for your specific commitments.",
            what_does_other_party_agree_to=s_data.get("what_does_other_party_agree_to") or s_data.get("counterparty_commitments") or "The other party agrees to perform the designated deliverables or obligations.",
            what_should_i_pay_attention_to=s_data.get("what_should_i_pay_attention_to") or s_data.get("attention_areas") or "Pay attention to notice periods, renewal conditions, and liability limits."
        )

        clauses = []
        raw_clauses = data.get("clauses", [])
        if isinstance(raw_clauses, list):
            for c in raw_clauses:
                if isinstance(c, dict):
                    clauses.append(ClauseItem(
                        clause_title=c.get("clause_title") or c.get("title") or "Key Provision",
                        category=c.get("category") or "Responsibilities",
                        what_the_document_says=c.get("what_the_document_says") or c.get("text") or "Stated in agreement text.",
                        in_simple_words=c.get("in_simple_words") or c.get("explanation") or "Establishes terms between parties.",
                        why_you_may_want_to_pay_attention=c.get("why_you_may_want_to_pay_attention") or c.get("impact") or "Affects operational or legal rights.",
                        source=c.get("source") or "Document body"
                    ))

        obligations = []
        raw_obs = data.get("obligations", [])
        if isinstance(raw_obs, list):
            for o in raw_obs:
                if isinstance(o, dict):
                    obligations.append(ObligationItem(
                        party=o.get("party") or "Your obligations",
                        obligation=o.get("obligation") or o.get("duty") or "Comply with agreement terms.",
                        deadline_or_timeline=o.get("deadline_or_timeline") or o.get("timeline") or "Not specified in the document.",
                        source=o.get("source") or "Document body"
                    ))

        attention_points = []
        raw_att = data.get("attention_points", [])
        if isinstance(raw_att, list):
            for a in raw_att:
                if isinstance(a, dict):
                    attention_points.append(AttentionPoint(
                        item=a.get("item") or a.get("title") or "Provisional Term",
                        observation=a.get("observation") or a.get("note") or "Review stated clauses.",
                        why_it_matters=a.get("why_it_matters") or a.get("impact") or "Has contractual implications.",
                        source=a.get("source") or "Document body"
                    ))

        checklist = []
        raw_chk = data.get("checklist", [])
        if isinstance(raw_chk, list):
            for ch in raw_chk:
                if isinstance(ch, dict):
                    checklist.append(ActionChecklistItem(
                        category=ch.get("category") or "Before Signing",
                        task=ch.get("task") or "Review terms with counterparty",
                        detail=ch.get("detail") or "Ensure mutual understanding of commitments.",
                        completed=bool(ch.get("completed", False))
                    ))

        lawyer_questions = []
        raw_q = data.get("lawyer_questions", [])
        if isinstance(raw_q, list):
            for q in raw_q:
                if isinstance(q, dict):
                    lawyer_questions.append(LawyerQuestion(
                        category=q.get("category") or "General Review",
                        question=q.get("question") or "What are the primary risks associated with this contract?",
                        context=q.get("context") or "Reviewing general provisions before execution."
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
