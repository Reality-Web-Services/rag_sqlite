"""Factory for creating vector stores."""
from typing import Optional, Dict, Any
from .base import VectorStore
from .sqlite import SQLiteVectorStore
from .milvus import MilvusVectorStore

class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create(store_type: str, **kwargs) -> VectorStore:
        """Create a vector store instance.
        
        Args:
            store_type: Type of vector store ('sqlite' or 'milvus')
            **kwargs: Arguments to pass to the vector store constructor
        
        Returns:
            VectorStore: An instance of the requested vector store
        """
        stores = {
            'sqlite': SQLiteVectorStore,
            'milvus': MilvusVectorStore
        }
        
        if store_type not in stores:
            raise ValueError(f"Unknown store type: {store_type}. Available types: {list(stores.keys())}")
            
        return stores[store_type](**kwargs)

def create_vector_store(
    store_type: str,
    config: Optional[Dict[str, Any]] = None
) -> VectorStore:
    """Convenience function to create a vector store with configuration.
    
    Args:
        store_type: Type of vector store ('sqlite' or 'milvus')
        config: Configuration for the vector store. If None, uses defaults.
            For SQLite: {'db_path': 'vectors.db'}
            For Milvus: {'collection_name': 'documents', 'host': 'localhost', 'port': 19530}
    
    Returns:
        VectorStore: Configured vector store instance
    """
    if config is None:
        config = {}
        
    # Set default configs
    defaults = {
        'sqlite': {'db_path': 'vectors.db'},
        'milvus': {'collection_name': 'documents', 'host': 'localhost', 'port': 19530}
    }
    
    # Merge provided config with defaults
    store_config = {**defaults.get(store_type, {}), **config}
    
    return VectorStoreFactory.create(store_type, **store_config)
