"""
Unit tests for BM25 Retriever
"""

import unittest
from app.core.retriever import BM25Retriever

class TestBM25Retriever(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.retriever = BM25Retriever()
        self.test_docs = [
            "Pasta carbonara is made with eggs and cheese",
            "Biryani requires rice and aromatic spices",
            "Chocolate cake needs cocoa powder and sugar"
        ]
        self.retriever.index_documents(self.test_docs)
    
    def test_tokenization(self):
        """Test tokenization"""
        tokens = self.retriever.tokenize("Hello World!")
        self.assertEqual(tokens, ['hello', 'world'])
    
    def test_indexing(self):
        """Test document indexing"""
        self.assertEqual(len(self.retriever.documents), 3)
        self.assertGreater(len(self.retriever.idf), 0)

    def test_retrieval(self):
        """Test BM25 retrieval"""
        results = self.retriever.retrieve("pasta eggs", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("carbonara", results[0][0].lower())
    
    def test_empty_query(self):
        """Test empty query handling"""
        results = self.retriever.retrieve("", top_k=1)
        self.assertIsNotNone(results)

if __name__ == '__main__':
    unittest.main()    