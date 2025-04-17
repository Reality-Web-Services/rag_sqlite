"""Compare RAG and LlamaIndex processors."""
import os
from dotenv import load_dotenv
from rag_sqlite.vectorstores.factory import create_vector_store
from rag_sqlite.core.rag_processor import RAGProcessor
from rag_sqlite.processors.llama_processor import LlamaProcessor

def test_llama_processor():
    """Test the LlamaIndex processor."""
    print("\n=== Testing LlamaIndex Processor ===")
    
    processor = LlamaProcessor(
        milvus_uri="http://localhost:19530",
        collection_name="documents_llama",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Add document
    pdf_path = "/home/goat/deve/RAGGIN_ONE/pdfs_to_rag/SuttonBartoIPRLBook2ndEd.pdf"
    doc_id = processor.add_document(
        pdf_path,
        metadata={
            "title": "Reinforcement Learning: An Introduction",
            "author": "Sutton et al",
            "processor": "llama"
        }
    )
    print(f"Added document with ID: {doc_id}")
    
    # Test query
    question = "What is the difference between policy and value iteration?"
    print(f"\nQuestion: {question}")
    
    result = processor.query(question)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        metadata = source.get('metadata', {})
        score = source.get('score', 0)
        print(f"\n- Score: {score:.2f if score else 'N/A'}")
        print(f"  {source['text'][:200]}...")

def test_rag_processor():
    """Test the original RAG processor."""
    print("\n=== Testing Original RAG Processor ===")
    
    # Initialize vector store and processor
    store_type = "milvus"
    store_config = {}
    vector_store = create_vector_store(store_type, store_config)
    
    processor = RAGProcessor(
        vector_store=vector_store,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Add document
    pdf_path = "/home/goat/deve/RAGGIN_ONE/pdfs_to_rag/SuttonBartoIPRLBook2ndEd.pdf"
    doc_ids = processor.add_document(
        pdf_path,
        metadata={
            "title": "Reinforcement Learning: An Introduction",
            "author": "Sutton et al",
            "processor": "rag"
        }
    )
    print(f"Added document with {len(doc_ids)} sections")
    
    # Test query
    question = "What is the difference between policy and value iteration?"
    print(f"\nQuestion: {question}")
    
    result = processor.query(question)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"\n- From {source['metadata']['header']} (score: {source['score']:.2f}):")
        print(f"  {source['text'][:200]}...")

def main():
    # Load environment variables
    load_dotenv()
    
    # Test both processors
    test_llama_processor()
    test_rag_processor()

if __name__ == "__main__":
    main()
