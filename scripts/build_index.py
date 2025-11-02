"""
Build BM25 index from raw recipes
Method 1: Keyword-based indexing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.retriever import BM25Retriever
from app.utils.text_processor import TextProcessor
from app.config import Config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def build_index():
    """Build BM25 index from raw recipes"""
    
    logger.info("="*70)
    logger.info("Building BM25 Index - Method 1")
    logger.info("="*70)
    
    # Initialize components
    text_processor = TextProcessor()
    retriever = BM25Retriever()

    # Load raw recipes
    logger.info(f"Loading recipes from {Config.RAW_RECIPES_PATH}")
    
    if not os.path.exists(Config.RAW_RECIPES_PATH):
        logger.error(f"Recipe file not found: {Config.RAW_RECIPES_PATH}")
        return
    
    with open(Config.RAW_RECIPES_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Chunk text
    logger.info("Chunking text...")
    chunks = text_processor.chunk_text(content)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Index documents with BM25
    logger.info("Indexing chunks with BM25...")
    retriever.index_documents(chunks)
    
    logger.info("="*70)
    logger.info("Index building complete!")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Unique tokens: {len(retriever.idf)}")
    logger.info("="*70)

if __name__ == "__main__":
    build_index()    