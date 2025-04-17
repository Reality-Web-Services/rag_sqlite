"""Script to ingest documents into the vector store."""
import os
import argparse
from dotenv import load_dotenv
from vectorstores.factory import create_vector_store
from core.rag_processor import RAGProcessor
from processors.llama_processor import LlamaProcessor

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--processor", 
        choices=["rag", "llama"], 
        default="llama",
        help="Which processor to use (default: llama)"
    )
    parser.add_argument(
        "--pdf", 
        type=str,
        default="/home/goat/deve/RAGGIN_ONE/pdfs_to_rag/SuttonBartoIPRLBook2ndEd.pdf",
        help="Path to PDF file to ingest"
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if args.processor == "llama":
        # Initialize LlamaIndex processor
        processor = LlamaProcessor(
            milvus_uri="http://localhost:19530",
            collection_name="documents",
            anthropic_api_key=anthropic_api_key
        )
        
        # Add document
        doc_id = processor.add_document(
            args.pdf,
            metadata={
                "title": "Reinforcement Learning: An Introduction",
                "author": "Sutton et al",
                "processor": "llama"
            }
        )
        print(f"Added document with ID: {doc_id}")
        
    else:
        # Get store type from env or use default
        store_type = os.getenv("VECTOR_STORE_TYPE", "milvus")
        store_config = {}
        
        # Initialize vector store and RAG processor
        vector_store = create_vector_store(store_type, store_config)
        processor = RAGProcessor(
            vector_store=vector_store,
            anthropic_api_key=anthropic_api_key
        )
        
        # Add document
        doc_ids = processor.add_document(
            args.pdf,
            metadata={
                "title": "Reinforcement Learning: An Introduction",
                "author": "Sutton et al",
                "processor": "rag"
            }
        )
        print(f"Added document with {len(doc_ids)} sections")

if __name__ == "__main__":
    main()
