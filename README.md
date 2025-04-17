# RAG System with Multiple Vector Stores

A Retrieval-Augmented Generation (RAG) system that supports multiple vector store backends:
- Milvus (default)
- SQLite

## Features
- Flexible vector store backend switching via environment variables
- Automatic text chunking and embedding
- PDF document processing
- Detailed API call logging with unique timestamps
- Claude-2 LLM integration

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Required
export ANTHROPIC_API_KEY=your_key_here

# Optional (defaults to Milvus)
export VECTOR_STORE_TYPE=milvus  # or 'sqlite'
```

## Usage

### Adding Documents
```bash
python -m rag_sqlite.ingest
```

### Querying
```bash
python -m rag_sqlite.query
```

## Vector Store Configuration

### Milvus
- Default host: localhost
- Default port: 19530
- Default collection: documents

### SQLite
- Default database: vectors.db
