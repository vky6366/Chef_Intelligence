import re
import math
from typing import List, Tuple
from collections import Counter
# from app.utils.logger import setup_logger
from app.config import Config

# logger = setup_logger(__name__)

class BM25Retriever:
    """
    BM25 (Best Match 25) keyword-based retrieval algorithm
    Method 1: Direct keyword search without vector embeddings
    """

    def __init__(self, k1: float = None, b: float = None):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1 or Config.BM25_K1
        self.b = b or Config.BM25_B
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf = {}
        self.tokenized_docs = []

        # logger.info(f"BM25 Retriever initialized (k1={self.k1}, b={self.b})")

    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split by non-alphanumeric
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index_documents(self, documents: List[str]):
        """
        Index documents for BM25 retrieval
        
        Args:
            documents: List of document strings
        """
        # logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # Calculate document frequencies
        self.doc_freqs = {}
        for tokenized_doc in self.tokenized_docs:
            unique_tokens = set(tokenized_doc)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Calculate IDF (Inverse Document Frequency)
        num_docs = len(self.documents)
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        
        # logger.info(f"Indexing complete. Unique tokens: {len(self.idf)}")

    def calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document given query tokens
        
        Args:
            query_tokens: List of query tokens
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            term_freq = doc_term_freqs.get(token, 0)
            idf_score = self.idf[token]

            # BM25 formula
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf_score * (numerator / denominator)
        
        return score                

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        query_tokens = self.tokenize(query)

        # Calculate scores for all documents
        scores = []
        for idx in range(len(self.documents)):
            score = self.calculate_bm25_score(query_tokens, idx)
            scores.append((idx, score))
        
        # Sort by score (descending) and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        
        # Return documents with scores
        results = [(self.documents[idx], score) for idx, score in top_results]
        
        # logger.debug(f"Retrieved {len(results)} documents for query: {query}")
        return results
