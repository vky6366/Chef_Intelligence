from typing import List
from app.config import Config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextProcessor:
    """
    Text processing utilities for Method 1
    Simple paragraph-based chunking
    """

    def __init__(self, chunk_size: int = None, overlap: int = None):
        """
        Initialize text processor
        
        Args:
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.overlap = overlap or Config.CHUNK_OVERLAP
        
        logger.info(f"Text Processor initialized (chunk_size={self.chunk_size}, overlap={self.overlap})")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraphs and size
        Method 1: Simple paragraph-based text breaking
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # First split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk_size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from the end of previous chunk
                words = current_chunk.split()
                overlap_text = ' '.join(words[-self.overlap:]) if len(words) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks
    
    def semantic_chunking(text, similarity_threshold=0.5):
        sentences = text.split('.')
        embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate cosine similarity between consecutive sentences
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            if similarity < similarity_threshold:
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        chunks.append('. '.join(current_chunk))
        return chunks