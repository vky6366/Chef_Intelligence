"""
End-to-end testing script for Method 1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.retriever import BM25Retriever
from app.core.llm_manager import LLMManager
from app.utils.text_processor import TextProcessor
from app.config import Config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_pipeline():
    """Run complete RAG pipeline"""
    
    print("="*70)
    print("🍳 CHEF INTELLIGENCE - METHOD 1 PIPELINE TEST")
    print("="*70)
    
    # Initialize components
    text_processor = TextProcessor()
    retriever = BM25Retriever()
    llm_manager = LLMManager()

    # Load and index recipes
    print("\n📚 Loading recipes...")
    with open(Config.RAW_RECIPES_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = text_processor.chunk_text(content)
    retriever.index_documents(chunks)
    print(f"✓ Indexed {len(chunks)} chunks")

    # Test queries
    test_queries = [
        "How to make pasta?",
        "What are the ingredients for biryani?",
        "Give me a chocolate cake recipe",
        "How to make stir fry vegetables?"
    ]
    
    print("\n🧪 Running test queries...\n")

    for query in test_queries:
        print("-"*70)
        print(f"Q: {query}")
        
        # Retrieve
        retrieved = retriever.retrieve(query, top_k=3)
        chunks_list = [chunk for chunk, score in retrieved]
        scores = [score for chunk, score in retrieved]
        
        print(f"✓ Retrieved {len(chunks_list)} chunks (BM25 scores: {[f'{s:.2f}' for s in scores]})")
        
        # Generate
        answer = llm_manager.generate_response(query, chunks_list)
        print(f"\nA: {answer}\n")
    
    print("="*70)
    print("✅ Pipeline test complete!")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()