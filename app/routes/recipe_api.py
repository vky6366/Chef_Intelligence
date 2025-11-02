from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from app.core.retriever import BM25Retriever
from app.core.llm_manager import get_llm_manager
from app.utils.text_processor import TextProcessor
from app.utils.logger import setup_logger
from app.config import Config
import os
import torch
import platform

router = APIRouter()
logger = setup_logger(__name__)

# Initialize components
text_processor = TextProcessor()

# Retriever will be initialized in startup (safer for Docker)
retriever: BM25Retriever = None

# LLM manager singleton
llm_manager = None

def get_or_init_llm():
    """Get or initialize LLM manager (singleton)"""
    global llm_manager
    if llm_manager is None:
        logger.info(f"Initializing TinyLlama (first request) on {platform.system()}...")
        llm_manager = get_llm_manager()
    return llm_manager

@router.on_event("startup")
async def startup_event():
    """Initialize retriever with recipe data safely"""
    global retriever
    try:
        retriever = BM25Retriever()  # Initialize retriever safely
        
        if os.path.exists(Config.RAW_RECIPES_PATH):
            with open(Config.RAW_RECIPES_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = text_processor.chunk_text(content)
            if chunks:
                retriever.index_documents(chunks)
                logger.info(f"✓ Indexed {len(chunks)} recipe chunks")
            else:
                logger.warning("No chunks generated from recipe file.")
        else:
            logger.warning(f"Recipe file not found: {Config.RAW_RECIPES_PATH}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        retriever = BM25Retriever()  # fallback empty retriever

# Request/Response Models
class RecipeQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RetrievedChunk(BaseModel):
    content: str
    score: float

class RecipeQueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    method: str = "BM25 Keyword Search"
    model: str = "TinyLlama-1.1B-Chat-v1.0"
    device: str = "unknown"

@router.post("/recipe/query", response_model=RecipeQueryResponse)
async def query_recipe(request: RecipeQueryRequest):
    """Query recipes using BM25 keyword search with TinyLlama generation"""
    global retriever
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Retrieve relevant chunks using BM25
        if retriever is None or not retriever.documents:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Retriever is not initialized or no documents indexed"
            )

        retrieved = retriever.retrieve(request.query, top_k=request.top_k)
        
        if not retrieved:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant recipes found"
            )
        
        chunks = [chunk for chunk, score in retrieved]
        scores = [score for chunk, score in retrieved]
        
        # Step 2: Generate response using TinyLlama
        llm = get_or_init_llm()
        answer = llm.generate_response(request.query, chunks)
        
        # Step 3: Prepare response
        retrieved_chunks = [
            RetrievedChunk(content=chunk, score=score)
            for chunk, score in zip(chunks, scores)
        ]
        
        return RecipeQueryResponse(
            query=request.query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            device=llm.device
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/recipe/stats")
async def get_stats():
    """Get retriever statistics"""
    global retriever
    return {
        "total_chunks": len(retriever.documents) if retriever else 0,
        "indexed_tokens": len(retriever.idf) if retriever else 0,
        "method": "BM25 Keyword Search",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "device": get_or_init_llm().device if llm_manager else "unknown",
        "accuracy": "~20%"
    }

@router.get("/model/info")
async def get_model_info():
    """Get model information"""
    llm = get_or_init_llm()
    return {
        "model_name": Config.LLM_MODEL_NAME,
        "device": llm.device,
        "max_new_tokens": Config.LLM_MAX_NEW_TOKENS,
        "temperature": Config.LLM_TEMPERATURE,
        "quantization": {
            "8bit": Config.LOAD_IN_8BIT,
            "4bit": Config.LOAD_IN_4BIT
        },
        "status": "loaded",
        "os": platform.system()
    }
