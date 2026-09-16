"""Retrieval-Augmented Generation (RAG) indexing and search for LegalLens."""

from typing import List, Dict, Any, Optional, Tuple
from app.models import DocumentChunk
from app.embeddings import TextVectorizer


class LegalRAGIndex:
    """Document vector index providing grounded snippet retrieval."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.vectorizer = TextVectorizer(ngram_range=(1, 2))
        self.chunk_vectors: List[Dict[str, float]] = []
        
        if self.chunks:
            texts = [c.text for c in self.chunks]
            self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 4, score_threshold: float = 0.04) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks with grounding metadata and relevance score.
        Returns empty list if no chunks meet the score threshold.
        """
        if not self.chunks or not query.strip():
            return []

        query_vec = self.vectorizer.transform(query)
        scored_results = []

        for idx, (chunk, c_vec) in enumerate(zip(self.chunks, self.chunk_vectors)):
            score = self.vectorizer.cosine_similarity(query_vec, c_vec)
            
            # Boost score if chunk section header matches query terms
            if chunk.section_title:
                header_tokens = set(chunk.section_title.lower().split())
                query_tokens = set(query.lower().split())
                if header_tokens.intersection(query_tokens):
                    score += 0.15

            if score >= score_threshold:
                # Format standard citation string
                citation_parts = [f"Page {chunk.page_number}"]
                if chunk.section_title:
                    citation_parts.append(chunk.section_title)
                citation = ", ".join(citation_parts)

                scored_results.append({
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "citation": citation,
                    "text": chunk.text,
                    "score": round(score, 4)
                })

        # Sort descending by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def get_grounded_context(self, query: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """
        Extracts concatenated grounded context and formatted citations for LLM prompt.
        """
        from typing import Tuple
        results = self.search(query, top_k=top_k)
        if not results:
            return "", []

        context_blocks = []
        citations = []
        for r in results:
            context_blocks.append(f"--- [Source: {r['citation']}] ---\n{r['text']}")
            citations.append(r['citation'])

        return "\n\n".join(context_blocks), citations
