"""Pydantic data schemas for LegalLens."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Chunk of document text with source provenance."""
    chunk_id: str
    page_number: int
    section_title: Optional[str] = None
    text: str
    char_start: int
    char_end: int


class DocumentMetadata(BaseModel):
    """Extracted high-level document attributes."""
    document_type: str = Field(description="Type of legal document e.g. Residential Lease, Employment Agreement, NDA")
    approximate_length: str = Field(description="e.g. 4 pages, ~1,800 words")
    parties_involved: str = Field(description="Identified parties or 'Not found in the document.'")
    effective_date: str = Field(description="Effective date or 'Not found in the document.'")
    expiration_date: str = Field(description="Expiration or renewal date or 'Not found in the document.'")
    governing_jurisdiction: str = Field(description="Governing law/jurisdiction or 'Not found in the document.'")


class PlainEnglishSummary(BaseModel):
    """Clear, plain-English summary answering core user questions."""
    what_is_this: str = Field(description="Plain-language description of what this document is and its purpose.")
    what_am_i_agreeing_to: str = Field(description="Concise breakdown of user's core commitments without legal jargon.")
    what_does_other_party_agree_to: str = Field(description="Concise breakdown of counterparty's commitments.")
    what_should_i_pay_attention_to: str = Field(description="Provisions deserving careful review.")


class ClauseItem(BaseModel):
    """Extracted meaningful clause with plain explanation and source."""
    clause_title: str = Field(description="Short, descriptive title of the clause.")
    category: str = Field(description="Category e.g. Payment, Term, Termination, Liability, Confidentiality, Renewal, etc.")
    what_the_document_says: str = Field(description="Concise factual extract/summary grounded directly in the text.")
    in_simple_words: str = Field(description="What this clause means in plain everyday language.")
    why_you_may_want_to_pay_attention: str = Field(description="Practical significance and implications.")
    source: str = Field(description="Source reference e.g. 'Page 2, Section 4 — Rent & Deposit'")


class ObligationItem(BaseModel):
    """Actionable responsibility or obligation."""
    party: str = Field(description="'Your obligations' or 'Other party's obligations'")
    obligation: str = Field(description="Specific actionable responsibility.")
    deadline_or_timeline: str = Field(description="Associated date, timeline, or 'Not specified in the document.'")
    source: str = Field(description="Source citation e.g. 'Section 3.2'")


class AttentionPoint(BaseModel):
    """Provisions that a reasonable user may want to review carefully (in neutral tone)."""
    item: str = Field(description="Short title e.g. 'Automatic 60-day renewal'")
    observation: str = Field(description="Neutral description of what the clause provides without alarmist language.")
    why_it_matters: str = Field(description="Practical real-world impact to be aware of.")
    source: str = Field(description="Document reference.")


class ActionChecklistItem(BaseModel):
    """Actionable checklist item for user before signing or acting."""
    category: str = Field(description="'Before Signing' or 'After Signing / Ongoing'")
    task: str = Field(description="Specific verification or action item.")
    detail: str = Field(description="Why this step matters based on document terms.")
    completed: bool = False


class LawyerQuestion(BaseModel):
    """Document-grounded question to ask a qualified legal professional or counterparty."""
    category: str = Field(description="e.g. 'Termination & Renewal', 'Financial Liability', 'Scope of Work'")
    question: str = Field(description="Practical, specific question derived from the document.")
    context: str = Field(description="Why asking this question is helpful based on the contract text.")


class DocumentAnalysis(BaseModel):
    """Full comprehensive legal document analysis payload."""
    metadata: DocumentMetadata
    summary: PlainEnglishSummary
    clauses: List[ClauseItem] = Field(default_factory=list)
    obligations: List[ObligationItem] = Field(default_factory=list)
    attention_points: List[AttentionPoint] = Field(default_factory=list)
    checklist: List[ActionChecklistItem] = Field(default_factory=list)
    lawyer_questions: List[LawyerQuestion] = Field(default_factory=list)
    disclaimer: str = (
        "LegalLens provides general information and document assistance, not legal advice. "
        "AI-generated explanations may be incomplete or incorrect. "
        "For important legal decisions or disputes, consult a qualified legal professional."
    )


class ComparisonChange(BaseModel):
    """Meaningful difference between two versions of a contract."""
    category: str = Field(description="Category e.g. Payment, Term, Notice Period, Liability, Renewal")
    what_changed: str = Field(description="High-level description of what altered between versions.")
    previous_version: str = Field(description="How the clause read or functioned in Document A.")
    new_version: str = Field(description="How the clause reads or functions in Document B.")
    what_this_could_mean: str = Field(description="Practical implications for the user in plain language.")
    source_previous: str = Field(description="Location in Document A e.g. 'Doc A, Page 1, Section 2'")
    source_new: str = Field(description="Location in Document B e.g. 'Doc B, Page 1, Section 2'")


class ComparisonResult(BaseModel):
    """Complete document comparison report."""
    document_a_name: str
    document_b_name: str
    summary_of_differences: str
    changes: List[ComparisonChange] = Field(default_factory=list)
    overall_takeaway: str
    disclaimer: str = (
        "LegalLens provides general information and document assistance, not legal advice. "
        "For important legal decisions or disputes, consult a qualified legal professional."
    )


class QAResponse(BaseModel):
    """Grounded question-and-answer response."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    grounded: bool = True
    disclaimer: str = "This answer is for informational purposes only and does not constitute legal advice."
