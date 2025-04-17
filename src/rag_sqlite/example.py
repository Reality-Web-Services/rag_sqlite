"""Example usage of the RAG system."""
import os
from dotenv import load_dotenv
from vectorstores.sqlite import SQLiteVectorStore
from vectorstores.milvus import MilvusVectorStore
from core.rag_processor import RAGProcessor

def main():
    # Load environment variables
    load_dotenv()
    
    # Choose your vector store
    # For SQLite:
    vector_store = SQLiteVectorStore(db_path="vectors.db")
    
    # For Milvus:
    # vector_store = MilvusVectorStore(
    #     collection_name="documents",
    #     host="localhost",
    #     port=19530
    # )
    
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
    
    # Query the system
    question = "What is the main topic of Chapter 1?"
    result = rag.query(question)
    
    print("\nQuestion:", question)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"\n- From {source['metadata']['header']} (score: {source['score']:.2f}):")
        print(f"  {source['text'][:200]}...")

if __name__ == "__main__":
    main()
