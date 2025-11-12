from typing import List, Callable, Optional
from app.config import Config
from app.utils.logger import setup_logger
import re
import numpy as np

logger = setup_logger(__name__)

class TextProcessor:
    """
    Text processing utilities.
    Two chunking strategies:
      - semantic_chunking: uses an encoder function to split by semantic breaks
      - char_chunking: fallback chunk by characters with overlap

    The encoder_fn should be a callable that accepts List[str] and returns
    numpy.ndarray or list-like of embeddings.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Args:
            chunk_size: approximate target chunk size in characters for char_chunking
            overlap: overlap in characters between adjacent chunks for char_chunking
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove weird characters.
        """
        if text is None:
            return ""
        # collapse multiple whitespace/newlines into single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter (works reasonably for recipe instructions).
        """
        # Keep the delimiter and strip spaces
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def char_chunking(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Deterministic chunk by characters with overlap (fallback).
        """
        if not text:
            return []
        cs = chunk_size or self.chunk_size
        ov = overlap or self.overlap
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + cs, n)
            chunks.append(text[start:end].strip())
            if end == n:
                break
            start = end - ov
            if start < 0:
                start = 0
        return [c for c in chunks if c]

    def semantic_chunking(self, text: str, encoder_fn: Callable[[List[str]], "np.ndarray"], similarity_threshold: float = 0.65, ) -> List[str]:
        """
        Chunk text using sentence-level semantic similarity.

        Steps:
          - split text to sentences
          - compute embeddings for sentences via encoder_fn
          - group consecutive sentences while consecutive similarity >= threshold

        Args:
          text: input string
          encoder_fn: function that takes List[str] and returns embeddings (np.ndarray or list)
          similarity_threshold: cosine similarity cutoff to keep sentences in same chunk

        Returns:
          list of chunk strings
        """
        text = self.clean_text(text)
        if not text:
            return []

        sents = self.split_into_sentences(text)
        if len(sents) == 0:
            return []

        # If only 1 sentence, return it
        if len(sents) == 1:
            return [sents[0]]

        # Get embeddings for sentences — ensure numpy array shape (N, D)
        emb = encoder_fn(sents)
        emb = np.asarray(emb)
        if emb.ndim != 2 or emb.shape[0] != len(sents):
            # fallback to char chunking if encoder returned strange shape
            return self.char_chunking(text)

        chunks = []
        current_chunk = [sents[0]]

        # pre-normalize vectors for cosine similarity
        norms = np.linalg.norm(emb, axis=1)
        # avoid division by zero
        norms[norms == 0] = 1e-8
        emb_normed = emb / norms[:, None]

        for i in range(1, len(sents)):
            sim = float(np.dot(emb_normed[i - 1], emb_normed[i]))
            if sim < similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sents[i]]
            else:
                current_chunk.append(sents[i])

        chunks.append(" ".join(current_chunk))

        # As a final safety: merge very small chunks with neighbors
        merged = []
        for c in chunks:
            if not merged:
                merged.append(c)
            elif len(c) < 40:  # tiny chunk, append to previous
                merged[-1] = merged[-1] + " " + c
            else:
                merged.append(c)

        return merged