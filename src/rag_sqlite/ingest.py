"""Script to ingest documents into the vector store."""
import os
from dotenv import load_dotenv
from vectorstores.factory import create_vector_store
from core.rag_processor import RAGProcessor

def main():
    # Load environment variables
    load_dotenv()
    
    # Get store type from env or use default
    store_type = os.getenv("VECTOR_STORE_TYPE", "milvus")
    store_config = {}
    
    # Initialize vector store
    vector_store = create_vector_store(store_type, store_config)
    
    # Initialize RAG processor
    rag = RAGProcessor(
        vector_store=vector_store,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Add a document
    pdf_path = "/home/goat/deve/RAGGIN_ONE/pdfs_to_rag/SuttonBartoIPRLBook2ndEd.pdf"
    doc_ids = rag.add_document(
        pdf_path,
        metadata={"title": "Reinforcement Learning: An Introduction", "author": "Sutton et al"}
    )
    print(f"Added document with {len(doc_ids)} sections")

if __name__ == "__main__":
    main()
