# RAG System Enhancement TODOs

## 1. LlamaIndex Integration
- [ ] Install new dependencies (llama-index and integrations)
- [ ] Create LlamaIndexProcessor class
  - [ ] Implement document loading with better chunking
  - [ ] Set up Milvus vector store integration
  - [ ] Configure sentence-transformers embeddings
  - [ ] Set up Claude LLM integration
- [ ] Add query engine configuration
- [ ] Implement response synthesizer

## 2. Structured Output Improvements
- [ ] Create Pydantic models for:
  - [ ] Document metadata
  - [ ] Query responses
  - [ ] Source citations
- [ ] Add response templates
- [ ] Implement structured output parsing

## 3. Query Enhancement
- [ ] Implement query planning
  - [ ] Add sub-query decomposition
  - [ ] Add query routing logic
- [ ] Add query templates
- [ ] Implement hybrid search (semantic + keyword)

## 4. Caching Layer
- [ ] Add SQLite cache for:
  - [ ] Document embeddings
  - [ ] Query results
  - [ ] LLM responses
- [ ] Implement cache invalidation
- [ ] Add document versioning

## 5. Evaluation & Monitoring
- [ ] Add evaluation metrics:
  - [ ] Answer relevance
  - [ ] Source quality
  - [ ] Response latency
- [ ] Implement structured logging
- [ ] Add performance monitoring
- [ ] Create debug mode with detailed tracing

## 6. Document Processing
- [ ] Enhance text extraction:
  - [ ] Better section detection
  - [ ] Table extraction
  - [ ] Image captions
- [ ] Add support for:
  - [ ] Markdown
  - [ ] HTML
  - [ ] Code files
- [ ] Improve metadata extraction

## Implementation Order
1. Set up LlamaIndex base integration
2. Enhance document processing
3. Add structured outputs
4. Implement caching
5. Add evaluation metrics
6. Enhance query capabilities
