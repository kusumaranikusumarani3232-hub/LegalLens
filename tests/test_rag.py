"""Tests for RAG Index and Grounding."""

import pytest
from app.models import DocumentChunk
from app.rag import LegalRAGIndex
from app.embeddings import TextVectorizer


def test_vectorizer_similarity():
    vec = TextVectorizer(ngram_range=(1, 2))
    docs = [
        "The tenant agrees to pay monthly rent on the first day.",
        "Confidential information shall be kept secret by receiving party.",
        "Either party may terminate upon sixty days written notice."
    ]
    vectors = vec.fit_transform(docs)
    assert len(vectors) == 3

    query_rent = vec.transform("monthly rent payment")
    score_rent_doc0 = TextVectorizer.cosine_similarity(query_rent, vectors[0])
    score_rent_doc1 = TextVectorizer.cosine_similarity(query_rent, vectors[1])
    
    # Query about rent should match rent document much higher than NDA document
    assert score_rent_doc0 > score_rent_doc1


def test_rag_grounded_retrieval():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            page_number=1,
            section_title="Rent and Payment",
            text="The tenant shall pay ₹25,000 monthly rent before the 5th of each month.",
            char_start=0,
            char_end=75
        ),
        DocumentChunk(
            chunk_id="c2",
            page_number=2,
            section_title="Termination",
            text="Either party may terminate this agreement with 30 days written notice.",
            char_start=76,
            char_end=150
        )
    ]
    rag = LegalRAGIndex(chunks)

    # Search for termination
    results = rag.search("notice required for termination", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c2"
    assert "Termination" in results[0]["citation"]
    assert results[0]["page_number"] == 2

    # Context formatting
    context, citations = rag.get_grounded_context("termination notice")
    assert len(citations) >= 1
    assert "Page 2" in citations[0]


def test_rag_unrelated_query_threshold():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            page_number=1,
            section_title="Premises",
            text="The apartment is located at outer ring road.",
            char_start=0,
            char_end=50
        )
    ]
    rag = LegalRAGIndex(chunks)
    # Search for something completely absent from document
    results = rag.search("quantum nuclear physics spacecraft satellite", top_k=2, score_threshold=0.2)
    assert len(results) == 0
