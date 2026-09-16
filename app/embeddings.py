"""Lightweight vectorization and semantic similarity calculations for LegalLens."""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple


class TextVectorizer:
    """Fast, lightweight TF-IDF and n-gram vectorizer with cosine similarity."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)):
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Cleans text into normalized alphanumeric tokens and n-grams."""
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [t for t in clean.split() if len(t) > 1]
        
        ngrams = []
        # Unigrams
        ngrams.extend(tokens)
        # Bigrams if specified
        if self.ngram_range[1] >= 2 and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                ngrams.append(f"{tokens[i]}_{tokens[i+1]}")
        return ngrams

    def fit_transform(self, documents: List[str]) -> List[Dict[str, float]]:
        """Fits vocabulary & IDF from documents and returns TF-IDF sparse vectors."""
        self.doc_count = len(documents)
        doc_tokens_list = [self._tokenize(doc) for doc in documents]
        
        # Calculate Document Frequency (DF)
        df_counter: Counter = Counter()
        for doc_tokens in doc_tokens_list:
            unique_tokens = set(doc_tokens)
            df_counter.update(unique_tokens)

        # Build vocabulary & IDF
        self.vocabulary = {term: idx for idx, term in enumerate(df_counter.keys())}
        self.idf = {
            term: math.log((self.doc_count + 1.0) / (df + 1.0)) + 1.0
            for term, df in df_counter.items()
        }

        # Compute TF-IDF for each document
        vectors = []
        for doc_tokens in doc_tokens_list:
            vectors.append(self._vectorize(doc_tokens))
        return vectors

    def transform(self, query: str) -> Dict[str, float]:
        """Transforms a query string into a normalized TF-IDF vector."""
        tokens = self._tokenize(query)
        return self._vectorize(tokens)

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total_tokens = max(1, len(tokens))
        vec: Dict[str, float] = {}
        
        norm_sq = 0.0
        for term, count in tf.items():
            if term in self.idf:
                weight = (count / total_tokens) * self.idf[term]
                vec[term] = weight
                norm_sq += weight * weight

        # L2 normalize
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for term in vec:
                vec[term] /= norm

        return vec

    @staticmethod
    def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Computes dot product between two normalized sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0
        # Iterate over the smaller dict
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
            
        dot_product = 0.0
        for term, weight_a in vec_a.items():
            if term in vec_b:
                dot_product += weight_a * vec_b[term]
        return dot_product
