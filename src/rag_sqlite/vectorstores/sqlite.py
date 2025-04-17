"""SQLite-based vector store implementation."""
import json
import sqlite3
import uuid
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

from .base import VectorStore

nltk.download('punkt', quiet=True)

class SQLiteVectorStore(VectorStore):
    """SQLite implementation using BM25 for similarity search."""
    
    def __init__(self, db_path: str = "vectors.db"):
        self.db_path = db_path
        self.setup_db()
        self.bm25_index = None
        self.documents = []
        self.load_documents()
    
    def setup_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                tokens TEXT
            )
            ''')
    
    def load_documents(self):
        """Load documents from SQLite and rebuild BM25 index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('SELECT * FROM documents').fetchall()
            
            self.documents = []
            tokenized_docs = []
            
            for row in rows:
                self.documents.append({
                    'id': row['id'],
                    'text': row['text'],
                    'metadata': json.loads(row['metadata']),
                    'tokens': json.loads(row['tokens'])
                })
                tokenized_docs.append(json.loads(row['tokens']))
            
            if tokenized_docs:
                self.bm25_index = BM25Okapi(tokenized_docs)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to the vector store."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        ids = []
        with sqlite3.connect(self.db_path) as conn:
            for text, metadata in zip(texts, metadatas):
                doc_id = str(uuid.uuid4())
                tokens = word_tokenize(text.lower())
                
                conn.execute(
                    'INSERT INTO documents (id, text, metadata, tokens) VALUES (?, ?, ?, ?)',
                    (doc_id, text, json.dumps(metadata), json.dumps(tokens))
                )
                ids.append(doc_id)
        
        self.load_documents()  # Rebuild index
        return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar texts using BM25."""
        if not self.bm25_index:
            return []
        
        query_tokens = word_tokenize(query.lower())
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k documents
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        
        for idx in top_k_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'id': doc['id'],
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'score': scores[idx]
                })
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """Delete texts by their IDs."""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' for _ in ids)
            conn.execute(f'DELETE FROM documents WHERE id IN ({placeholders})', ids)
        
        self.load_documents()  # Rebuild index
