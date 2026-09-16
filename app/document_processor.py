"""Document processing and text extraction for LegalLens."""

import io
import re
from typing import List, Dict, Any, Tuple, Optional
from pypdf import PdfReader
import docx

from app.models import DocumentChunk
from app.config import config


class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class DocumentProcessor:
    """Extracts, parses, and chunks legal documents (PDF, DOCX, TXT)."""

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extracts structured text from raw file bytes.
        Returns:
            - full_text: full concatenated text
            - pages_data: list of dicts with {"page_number": int, "text": str}
            - metadata: dict with document stats (page_count, word_count, char_count, is_scanned)
        """
        if not file_bytes or len(file_bytes) == 0:
            raise DocumentProcessingError("The uploaded file is empty.")

        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            return cls._extract_pdf(file_bytes)
        elif ext in ["docx", "doc"]:
            return cls._extract_docx(file_bytes)
        elif ext in ["txt", "text", "md"]:
            return cls._extract_txt(file_bytes)
        else:
            raise DocumentProcessingError(
                f"Unsupported file format (.{ext}). LegalLens supports PDF (.pdf), Word (.docx), and Plain Text (.txt)."
            )

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise DocumentProcessingError(f"Failed to read PDF file. The file may be damaged or corrupted. ({str(e)})")

        pages_data = []
        total_chars = 0
        total_pages = len(reader.pages)

        if total_pages == 0:
            raise DocumentProcessingError("The PDF contains no pages.")

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            clean_text = page_text.strip()
            total_chars += len(clean_text)
            pages_data.append({"page_number": page_num, "text": clean_text})

        # Detect scanned PDF (e.g. multi-page with less than 50 total characters extracted)
        is_scanned = (total_chars < 50 and total_pages > 0)

        full_text = "\n\n".join(
            f"--- [Page {p['page_number']}] ---\n{p['text']}" for p in pages_data if p['text']
        )

        words = full_text.split()
        metadata = {
            "page_count": total_pages,
            "char_count": total_chars,
            "word_count": len(words),
            "is_scanned": is_scanned,
            "format": "PDF"
        }

        if is_scanned:
            raise DocumentProcessingError(
                "This PDF appears to be a scanned image with no extractable computer-readable text. "
                "Please upload a text-based PDF, DOCX, or TXT document."
            )

        if not full_text.strip():
            raise DocumentProcessingError("No extractable text was found in the PDF.")

        return full_text, pages_data, metadata

    @classmethod
    def _extract_docx(cls, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise DocumentProcessingError(f"Failed to read Word (.docx) document. ({str(e)})")

        paragraphs = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                # Check heading style
                if p.style and "heading" in p.style.name.lower():
                    paragraphs.append(f"## {text}")
                else:
                    paragraphs.append(text)

        # Include table text if any
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n\n".join(paragraphs)
        if not full_text.strip():
            raise DocumentProcessingError("The DOCX document contains no readable text.")

        # Estimate page count for DOCX (~400 words per page standard)
        words = full_text.split()
        estimated_pages = max(1, (len(words) + 399) // 400)

        # Map paragraphs into estimated pages
        pages_data = []
        para_per_page = max(1, len(paragraphs) // estimated_pages)
        for i in range(estimated_pages):
            start = i * para_per_page
            end = (i + 1) * para_per_page if i < estimated_pages - 1 else len(paragraphs)
            page_content = "\n\n".join(paragraphs[start:end])
            pages_data.append({"page_number": i + 1, "text": page_content})

        metadata = {
            "page_count": estimated_pages,
            "char_count": len(full_text),
            "word_count": len(words),
            "is_scanned": False,
            "format": "DOCX"
        }

        return full_text, pages_data, metadata

    @classmethod
    def _extract_txt(cls, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        # Try UTF-8 first, then fallback to latin-1
        try:
            full_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                full_text = file_bytes.decode("latin-1")
            except Exception as e:
                raise DocumentProcessingError(f"Could not decode text file: {str(e)}")

        full_text = full_text.strip()
        if not full_text:
            raise DocumentProcessingError("The text file is empty.")

        words = full_text.split()
        estimated_pages = max(1, (len(words) + 399) // 400)

        # Check if text already has page markers e.g. [Page X]
        page_splits = re.split(r'(?:^|\n)---\s*\[Page\s+(\d+)\]\s*---', full_text, flags=re.IGNORECASE)
        pages_data = []

        if len(page_splits) > 2:
            # Document had explicit page markers
            for i in range(1, len(page_splits), 2):
                p_num = int(page_splits[i])
                p_txt = page_splits[i + 1].strip()
                pages_data.append({"page_number": p_num, "text": p_txt})
        else:
            # Synthesize pages
            lines = full_text.split("\n")
            lines_per_page = max(1, len(lines) // estimated_pages)
            for i in range(estimated_pages):
                start = i * lines_per_page
                end = (i + 1) * lines_per_page if i < estimated_pages - 1 else len(lines)
                page_content = "\n".join(lines[start:end]).strip()
                pages_data.append({"page_number": i + 1, "text": page_content})

        metadata = {
            "page_count": len(pages_data),
            "char_count": len(full_text),
            "word_count": len(words),
            "is_scanned": False,
            "format": "TXT"
        }

        return full_text, pages_data, metadata

    @classmethod
    def chunk_document(cls, pages_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Splits pages into bounded, overlapping chunks while preserving:
        - Page number provenance
        - Detected section headers
        - Word boundaries
        """
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        section_regex = re.compile(
            r'^(?:SECTION|ARTICLE|CLAUSE|\d+[\.\)]|[A-Z\s]{4,}:)\s*([^\n]+)',
            re.IGNORECASE | re.MULTILINE
        )

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]
            if not page_text.strip():
                continue

            # Identify section headers on this page
            current_section = None
            found_sections = section_regex.findall(page_text)
            if found_sections:
                current_section = found_sections[0].strip()[:60]

            # Break into paragraphs first
            paragraphs = page_text.split("\n\n")
            current_chunk_text = ""
            current_start_char = 0

            for p in paragraphs:
                p_str = p.strip()
                if not p_str:
                    continue

                # Check if this paragraph contains a section title
                header_match = section_regex.match(p_str)
                if header_match:
                    current_section = header_match.group(0).strip()[:60]

                if len(current_chunk_text) + len(p_str) < config.CHUNK_SIZE:
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + p_str
                    else:
                        current_chunk_text = p_str
                else:
                    # Flush current chunk
                    if current_chunk_text:
                        chunk_idx += 1
                        chunks.append(DocumentChunk(
                            chunk_id=f"chunk_{chunk_idx}",
                            page_number=page_num,
                            section_title=current_section,
                            text=current_chunk_text,
                            char_start=current_start_char,
                            char_end=current_start_char + len(current_chunk_text)
                        ))
                        current_start_char += len(current_chunk_text)

                    # Handle case where single paragraph exceeds CHUNK_SIZE
                    if len(p_str) >= config.CHUNK_SIZE:
                        # Split by words
                        words = p_str.split(" ")
                        sub_chunk = ""
                        for w in words:
                            if len(sub_chunk) + len(w) + 1 < config.CHUNK_SIZE:
                                sub_chunk = (sub_chunk + " " + w).strip()
                            else:
                                chunk_idx += 1
                                chunks.append(DocumentChunk(
                                    chunk_id=f"chunk_{chunk_idx}",
                                    page_number=page_num,
                                    section_title=current_section,
                                    text=sub_chunk,
                                    char_start=current_start_char,
                                    char_end=current_start_char + len(sub_chunk)
                                ))
                                current_start_char += len(sub_chunk)
                                sub_chunk = w
                        current_chunk_text = sub_chunk
                    else:
                        current_chunk_text = p_str

            if current_chunk_text:
                chunk_idx += 1
                chunks.append(DocumentChunk(
                    chunk_id=f"chunk_{chunk_idx}",
                    page_number=page_num,
                    section_title=current_section,
                    text=current_chunk_text,
                    char_start=current_start_char,
                    char_end=current_start_char + len(current_chunk_text)
                ))

        return chunks
