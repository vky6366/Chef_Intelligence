# Chef_Intelligence

AGI Chef Intelligence - Where AI meets flavour and precision.

---

## Methods Overview

### Method 1 – Direct RAG with Keyword Search (Accuracy: 20%)
**Implementation Details:**
- **Chunking Strategy:** Simple paragraph-based text breaking  
- **Retrieval Method:** BM25 keyword matching  
- **Vector Store:** None (direct text search)  
- **Processing:** Basic text splitting without semantic understanding  

### Method 2 – Semantic Chunking + Qdrant (Accuracy: 45%)
**Implementation Details:**
- **Chunking Strategy:** Intelligent semantic chunking  
- **Retrieval Method:** Semantic vector search with Sentence Transformers  
- **Vector Store:** Qdrant with 384-dimensional embeddings  
- **Model:** Snowflake/snowflake-arctic-embed-s (32M parameters)  

### Method 3 – Metadata Aware Retrieval (Accuracy: 60%)
**Implementation Details:**
- Integrates cuisine metadata for filtered retrieval  
- Uses dietary/type metadata for refined search  

### Method 4 – Agenetic Autonomous Search (Accuracy: 70%)
**Implementation Details:**
- **Multi-iteration Search:** System autonomously refines search strategies  
- **Adaptive Strategy:** Different approaches based on initial results  
- **Quality Evaluation:** Automatic assessment of result quality  
- **Autonomous Decision Making:** System decides when to continue searching  

### Method 5 – Persistent Reinforcement Memory (Accuracy: 80%)
**Implementation Details:**
- Implements long-term personalization using vector store retrieval memory  

### Method 6 – Hybrid RAG + Vector Store Retrieval Memory Fusion (Accuracy: 90%)
**Implementation Details:**
- Fuses live retrieval and persistent memory for full contextual coherence  

---

## Project Structure

chef_intelligence/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI entry point
│ ├── config.py # Configuration
│ ├── routes/
│ │ ├── init.py
│ │ └── recipe_api.py # Recipe endpoints
│ ├── core/
│ │ ├── init.py
│ │ ├── retriever.py # BM25 Retriever
│ │ └── llm_manager.py # LLM integration
│ └── utils/
│ ├── init.py
│ ├── text_processor.py # Text chunking
│ ├── prompt_builder.py # Prompt construction
│ └── logger.py # Logging system
├── data/
│ ├── raw_recipes/recipes.txt
│ ├── processed_chunks/
│ └── logs/
├── scripts/
│ ├── build_index.py # Build BM25 index
│ └── run_pipeline.py # Test pipeline
├── tests/
│ └── test_retriever.py
├── templates/
│ └── base_prompt.txt
├── requirements.txt
├── .env
└── README.md

## Setup Instructions

1. **Install dependencies:**
```bash
pip install -r requirements.txt

# Edit .env and add our API_KEY
cp .env.example .env

# Buld docker-compose
'''first open docker app and then go to the vs code terminal and typr given commsnd'''

docker-compose up --build

# CURL
'''
curl -X 'POST' \
  'http://0.0.0.0:8000/api/v1/recipe/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "chicken biryani recipe",
  "top_k": 3
}'
'''
