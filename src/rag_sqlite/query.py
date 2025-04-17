"""Script to query the RAG system."""
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
    
    # Interactive query loop
    print("\nRAG Query System (Ctrl+C to exit)")
    print("=" * 50)
    
    try:
        while True:
            # Get question from user
            question = input("\nEnter your question: ")
            if not question.strip():
                continue
                
            # Query the system
            result = rag.query(question)
            
            # Display results
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(f"\n- From {source['metadata']['header']} (score: {source['score']:.2f}):")
                print(f"  {source['text'][:200]}...")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
