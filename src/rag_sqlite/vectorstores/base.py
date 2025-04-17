"""Base classes for vector stores."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete texts by their IDs."""
        pass
