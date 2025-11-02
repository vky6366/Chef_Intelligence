from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import Config
from app.routes import recipe_api
from app.utils.logger import setup_logger
import os
import uvicorn

# Initialize logger
logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# Ensure required directories and recipe file exist
# -----------------------------------------------------------------------------
RECIPE_DIR = "data/raw_recipes"
RECIPE_FILE = os.path.join(RECIPE_DIR, "recipes.txt")

os.makedirs(RECIPE_DIR, exist_ok=True)

if not os.path.exists(RECIPE_FILE):
    logger.warning(f"Recipe file not found: {RECIPE_FILE}. Creating an empty file.")
    with open(RECIPE_FILE, "w", encoding="utf-8") as f:
        f.write("# Chef Intelligence Recipe Database\n")
        f.write("# Add your recipes here in plain text format.\n")

# -----------------------------------------------------------------------------
# Initialize FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Chef Intelligence - Method 1",
    description="Direct RAG with Keyword Search (BM25)",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
app.include_router(recipe_api.router, prefix="/api/v1", tags=["recipes"])

# -----------------------------------------------------------------------------
# Root endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Chef Intelligence - Method 1",
        "version": "1.0.0",
        "method": "Direct RAG with Keyword Search",
        "accuracy": "20%"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    """Run the FastAPI application"""
    logger.info("Starting Chef Intelligence - Method 1")
    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        workers=1  # Ensures PyTorch works safely inside Docker
    )

if __name__ == "__main__":
    main()
