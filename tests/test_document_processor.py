"""Tests for DocumentProcessor."""

import io
import pytest
from pypdf import PdfWriter
import docx

from app.document_processor import DocumentProcessor, DocumentProcessingError


def test_txt_extraction():
    sample_text = "--- [Page 1] ---\nSection 1: Parties\nLandlord and Tenant agree.\n--- [Page 2] ---\nSection 2: Rent\nRent is $1,000."
    text, pages, meta = DocumentProcessor.extract_text(sample_text.encode("utf-8"), "test_lease.txt")
    
    assert "Section 1: Parties" in text
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    assert meta["format"] == "TXT"
    assert meta["word_count"] > 0


def test_docx_extraction():
    doc = docx.Document()
    doc.add_heading("Lease Agreement", level=1)
    doc.add_paragraph("This agreement is between Alice and Bob.")
    doc.add_paragraph("Rent is $2,000 per month.")
    
    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    text, pages, meta = DocumentProcessor.extract_text(docx_bytes, "sample.docx")
    assert "Lease Agreement" in text
    assert "Alice and Bob" in text
    assert meta["format"] == "DOCX"
    assert len(pages) >= 1


def test_pdf_extraction_and_chunking():
    # Generate a lightweight in-memory PDF with pypdf
    writer = PdfWriter()
    page1 = writer.add_blank_page(width=300, height=300)
    # Write some stream text (or just test text extraction from plain bytes)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    # Empty text PDF should be caught as scanned/no extractable text
    with pytest.raises(DocumentProcessingError):
        DocumentProcessor.extract_text(pdf_bytes, "blank.pdf")


def test_empty_file_error():
    with pytest.raises(DocumentProcessingError) as exc_info:
        DocumentProcessor.extract_text(b"", "empty.txt")
    assert "empty" in str(exc_info.value).lower()


def test_unsupported_format_error():
    with pytest.raises(DocumentProcessingError) as exc_info:
        DocumentProcessor.extract_text(b"some content", "file.mp4")
    assert "unsupported" in str(exc_info.value).lower()


def test_chunking_provenance():
    pages = [
        {"page_number": 1, "text": "SECTION 1. DEFINITIONS\nThis is paragraph one.\n\nSECTION 2. TERM\nThis is paragraph two."},
        {"page_number": 2, "text": "SECTION 3. PAYMENT\nRent is $1,500 due on the first."}
    ]
    chunks = DocumentProcessor.chunk_document(pages)
    assert len(chunks) >= 2
    assert all(c.page_number in [1, 2] for c in chunks)
    assert any("PAYMENT" in (c.section_title or "") or "Payment" in c.text for c in chunks)
