"""Tests for grounded Q&A refusal and hallucination prevention."""

from app.llm_provider import LLMProvider
from app.prompts import QA_PROMPT


def test_qa_absent_fact_refusal():
    llm = LLMProvider(provider="mock")
    # Context contains only lease information about rent
    context = "--- [Source: Page 1, Section 2] ---\nThe monthly rent is ₹25,000 payable by the 5th of each month."
    
    # User asks an ungrounded question about pets or satellites
    prompt = QA_PROMPT.format(
        context=context,
        question="Can I keep three Siberian tigers and an eagle in the apartment?"
    )
    answer = llm.generate(prompt)

    # Must refuse to hallucinate
    assert "couldn't find this information in the uploaded document" in answer.lower()


def test_qa_grounded_answer():
    llm = LLMProvider(provider="mock")
    context = "--- [Source: Page 2, Section 3 — Security Deposit] ---\nThe security deposit of ₹1,00,000 shall be refunded within 21 business days."
    prompt = QA_PROMPT.format(
        context=context,
        question="What is the security deposit amount and refund timeline?"
    )
    answer = llm.generate(prompt)

    assert "security deposit" in answer.lower()
    assert "Page 2" in answer


def test_malformed_json_cleaning():
    llm = LLMProvider(provider="mock")
    # Markdown enclosed JSON with trailing comma
    bad_json = """```json
    {
      "key": "value",
      "number": 42,
    }
    ```"""
    parsed = llm._clean_and_parse_json(bad_json)
    assert parsed["key"] == "value"
    assert parsed["number"] == 42
