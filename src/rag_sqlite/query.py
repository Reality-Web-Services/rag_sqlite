"""Script to query the RAG system."""
import os
import argparse
from dotenv import load_dotenv
from vectorstores.factory import create_vector_store
from core.rag_processor import RAGProcessor
from processors.llama_processor import LlamaProcessor

def format_sources(sources):
    """Format source documents for display."""
    for source in sources:
        text = source["text"]
        metadata = source["metadata"]
        score = source["score"]
        
        # Get title from metadata or use filename
        title = metadata.get("title", metadata.get("file_name", "Unknown Source"))
        
        # Format score
        score_str = f"{score:.2f}" if score is not None else "N/A"
        
        print(f"\n- From {title} (score: {score_str}):")  
        print(f"  {text}\n")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument(
        "--processor", 
        choices=["rag", "llama"], 
        default="llama",
        help="Which processor to use (default: llama)"
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Initialize processor
    if args.processor == "llama":
        processor = LlamaProcessor(
            milvus_uri="http://localhost:19530",
            collection_name="documents",
            anthropic_api_key=anthropic_api_key
        )
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
    
    # Interactive query loop
    print(f"\nRAG Query System using {args.processor.upper()} processor (Ctrl+C to exit)")
    print("=" * 50)
    
    try:
        while True:
            # Get question from user
            question = input("\nEnter your question: ")
            if not question.strip():
                continue
                
            # Query the system
            result = processor.query(question)
            
            # Display results
            print("\nAnswer:", result["answer"])
            format_sources(result["sources"])
            
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
