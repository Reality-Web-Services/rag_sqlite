"""LlamaIndex-based document processor."""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.node_parser import SentenceSplitter

class LlamaProcessor:
    """Process documents using LlamaIndex."""
    
    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        anthropic_api_key: Optional[str] = None,
        reset_collection: bool = False
    ):
        """Initialize the LlamaIndex processor."""
        # Initialize Milvus connection
        from pymilvus import connections, Collection, utility
        
        # Connect to Milvus
        connections.connect(uri=milvus_uri)
        
        # Handle collection
        if reset_collection and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # Set up vector store with basic fields
        self.vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=collection_name,
            dim=768,  # Dimension for all-mpnet-base-v2
            overwrite=reset_collection
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Set up embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model
        )
        
        # Set up LLM
        llm = Anthropic(
            api_key=anthropic_api_key,
            model="claude-3-opus-20240229"
        )
        
        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        # Initialize or load the index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context,
        )
    
    def add_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the index.
        Returns the document ID.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load and process document using reader
        from llama_index.readers.file import PDFReader
        reader = PDFReader()
        documents = reader.load_data(file_path)
        
        # Add metadata to documents
        for doc in documents:
            # Convert metadata to JSON-serializable format
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                'file_path': str(file_path),
                'file_name': path.name,
                'page_number': doc.metadata.get('page_number', 0)
            })
            doc.metadata = doc_metadata
        
        # Insert into index
        nodes = self.index.storage_context.node_parser.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)
        
        # Return first document ID as reference
        return documents[0].doc_id if documents else ""
    
    def query(
        self,
        query_text: str,
        similarity_top_k: int = 3,
        response_mode: str = "compact"
    ) -> Dict[str, Any]:
        """
        Query the document index.
        Returns dict with answer and source nodes.
        """
        try:
            # Create query engine with specific parameters
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode=response_mode,
                streaming=False,
                node_postprocessors=[],  # Disable any default postprocessors
            )
            
            # Execute query
            response = query_engine.query(query_text)
            
            # Format response
            result = {
                "answer": response.response,
                "sources": []
            }
            
            # Process source nodes
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source = {
                        "text": node.text,
                        "score": float(node.score) if hasattr(node, 'score') else None,
                        "metadata": {}
                    }
                    
                    # Clean and format metadata
                    if hasattr(node, 'metadata'):
                        for k, v in node.metadata.items():
                            if v is not None:
                                source["metadata"][k] = str(v)
                    
                    result["sources"].append(source)
            
            return result
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }
