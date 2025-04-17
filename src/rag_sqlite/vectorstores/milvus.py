"""Milvus-based vector store implementation."""
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from sentence_transformers import SentenceTransformer

from .base import VectorStore

class MilvusVectorStore(VectorStore):
    """Milvus implementation using sentence transformers for embeddings."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 19530,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # Connect to Milvus
        connections.connect(host=host, port=port)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        schema = CollectionSchema(fields=fields, description="Document store")
        self.collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index("embedding", index_params)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to Milvus."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Process texts in chunks if they're too large
        MAX_TEXT_LENGTH = 65000  # Leave some buffer from the 65535 limit
        processed_texts = []
        processed_metadatas = []
        
        for text, metadata in zip(texts, metadatas):
            if len(text) > MAX_TEXT_LENGTH:
                # Split the text into chunks
                chunks = [text[i:i + MAX_TEXT_LENGTH] for i in range(0, len(text), MAX_TEXT_LENGTH)]
                for i, chunk in enumerate(chunks):
                    processed_texts.append(chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = i
                    chunk_metadata['total_chunks'] = len(chunks)
                    processed_metadatas.append(chunk_metadata)
            else:
                processed_texts.append(text)
                processed_metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.model.encode(processed_texts)
        
        # Generate IDs
        ids = [str(i) for i in range(len(processed_texts))]
        
        # Insert into Milvus
        entities = [
            ids,
            processed_texts,
            processed_metadatas,
            embeddings.tolist()
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
        return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts in Milvus."""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in Milvus
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["id", "text", "metadata"]
        )
        
        # Format results
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                'id': hit.entity.get('id'),
                'text': hit.entity.get('text'),
                'metadata': hit.entity.get('metadata'),
                'score': float(hit.score)
            })
        
        return formatted_results
    
    def delete(self, ids: List[str]) -> None:
        """Delete texts by their IDs."""
        expr = f'id in {ids}'
        self.collection.delete(expr)
        self.collection.flush()
