import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Config:
    """Configuration for Chef Intelligence with TinyLlama"""

    # API Configuration
    # ========================
    TINY_LLAMA_API_KEY = os.getenv("TINY_LLAMA_API_KEY")

    if not TINY_LLAMA_API_KEY:
        raise ValueError("Missing TINY_LLAMA_API_KEY in environment variables. Please check your .env file.")


    # LLM Configuration
    # ========================
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    LLM_DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
    LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", 512))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.95))
    LLM_TOP_K = int(os.getenv("LLM_TOP_K", 50))
    LLM_DO_SAMPLE = os.getenv("LLM_DO_SAMPLE", "true").lower() == "true"

  
    # Model Loading
    # ========================
    LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"


    # BM25 Configuration (Method 1)
    # ========================
    BM25_K1 = float(os.getenv("BM25_K1", 1.5))
    BM25_B = float(os.getenv("BM25_B", 0.75))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 3))


    # Chunking Configuration
    # ========================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


    # Data Paths
    # ========================
    RAW_RECIPES_PATH = os.getenv("RAW_RECIPES_PATH", "data/raw_recipes/recipes.txt")
    PROCESSED_CHUNKS_PATH = os.getenv("PROCESSED_CHUNKS_PATH", "data/processed_chunks")
    LOGS_PATH = os.getenv("LOGS_PATH", "data/logs")
    
    # Model cache directory - uses environment variable for Docker compatibility
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", 
                                 os.getenv("TRANSFORMERS_CACHE", "models/cache"))

    # Ensure cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    
    # Server Configuration
    # ========================
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"


    # Logging Configuration
    # ========================
    LOGGER_NAME = os.getenv("LOGGER_NAME", "chef_intelligence")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "data/logs/app.log")

    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOG_LEVEL)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(LOG_LEVEL)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    # Log Format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Prevent duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info("✅ Chef Intelligence Configuration Loaded Successfully")
    logger.info(f"Model: {LLM_MODEL_NAME} | Device: {LLM_DEVICE}")
    logger.info(f"Model Cache: {MODEL_CACHE_DIR}")
    logger.info(f"Server: {HOST}:{PORT}")


# Export a config instance for imports
config = Config()