"""Core RAG processor implementation."""
from typing import List, Dict, Any, Optional, Type
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import json
from datetime import datetime
from datetime import timezone
import os

from vectorstores.base import VectorStore
from processors.text_processor import TextProcessor

class RAGProcessor:
    """Main RAG processor that combines vector store and LLM."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        anthropic_api_key: str,
        model_name: str = "claude-2"
    ):
        self.vector_store = vector_store
        self.llm = ChatAnthropic(api_key=anthropic_api_key, model_name=model_name)
        self.text_processor = TextProcessor()
        
        # Set up the QA chain
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert in reinforcement learning. Based on the following excerpts from a textbook, please answer the question accurately and concisely. If you cannot answer the question based on the excerpts, say so.

Excerpts from textbook:
{context}

Question: {question}

Answer: Let me analyze the excerpts and provide a clear answer."""
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
    
    def add_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Process and add a document to the vector store."""
        sections = self.text_processor.process_document(file_path)
        texts = [section["content"] for section in sections]
        
        if metadata is None:
            metadata = {}
        
        # Add file metadata to each section
        metadatas = []
        for section in sections:
            section_metadata = metadata.copy()
            section_metadata.update({
                "file_path": file_path,
                "header": section["header"],
                "start_page": section["start_page"]
            })
            metadatas.append(section_metadata)
        
        return self.vector_store.add_texts(texts, metadatas)
    
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Query the RAG system."""
        # Get relevant documents
        results = self.vector_store.similarity_search(question, k=k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
        # Format context from retrieved documents
        context = "\n\n".join(f"[{r['metadata'].get('header', 'Section')}]\n{r['text']}" 
                            for r in results)
        
        # Get answer from LLM with callback to log API calls
        with get_openai_callback() as cb:
            answer = self.qa_chain.run(context=context, question=question)
            
            # Log to file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, '..', '..', '..', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            # Get store type and timestamp for unique filename
            store_type = self.vector_store.__class__.__name__.lower().replace('vectorstore', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"api_call_{store_type}_{timestamp}.jsonl")
            print(f"\nLogging API call to: {log_file}")
            
            api_call = {
                'timestamp': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                'question': question,
                'prompt': self.qa_prompt.format(context=context, question=question),
                'answer': answer,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_tokens': cb.total_tokens,
                'total_cost': cb.total_cost
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(api_call) + '\n')
        
        return {
            "answer": answer,
            "sources": [
                {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": r["score"]
                }
                for r in results
            ]
        }
