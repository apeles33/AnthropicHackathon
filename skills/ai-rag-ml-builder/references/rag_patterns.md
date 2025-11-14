# RAG Patterns and Best Practices

This document provides patterns and best practices for building Retrieval-Augmented Generation (RAG) systems.

## Core RAG Architecture

### Basic RAG Pipeline

1. **Indexing Phase**
   - Document ingestion and preprocessing
   - Text chunking with overlap
   - Embedding generation
   - Storage in vector database

2. **Retrieval Phase**
   - Query embedding
   - Similarity search
   - Result ranking and filtering

3. **Generation Phase**
   - Context assembly
   - Prompt construction
   - Response generation (simulated in demos)

## Document Processing Patterns

### Chunking Strategies

**Fixed-Size Chunking**
```
- Chunk size: 500-1000 words
- Overlap: 50-100 words
- Use case: General documents
- Pros: Simple, predictable
- Cons: May split semantic units
```

**Semantic Chunking**
```
- Split by paragraphs/sections
- Preserve logical boundaries
- Use case: Structured documents
- Pros: Better context preservation
- Cons: Variable chunk sizes
```

**Sentence-Window Chunking**
```
- 3-5 sentences per chunk
- Overlap by 1-2 sentences
- Use case: Q&A systems
- Pros: Natural boundaries
- Cons: Smaller context windows
```

### Metadata Enrichment

Always include:
- Source document ID
- Chunk position
- Timestamp
- Document category
- Author/source (when available)

Optional metadata:
- Section headers
- Keywords
- Entity mentions
- Summary

## Retrieval Patterns

### Hybrid Search

Combine multiple retrieval methods:
- **Vector search**: Semantic similarity
- **Keyword search**: Exact matches
- **Metadata filtering**: Category, date, etc.
- **Reranking**: Boost by freshness, popularity

### Query Enhancement

**Query Expansion**
- Add synonyms
- Include related terms
- Use domain vocabulary

**Query Decomposition**
- Break complex queries into sub-queries
- Retrieve for each part
- Combine results

**Hypothetical Document Embeddings (HyDE)**
- Generate hypothetical answer
- Embed and search with it
- Often finds better matches

## Vector Database Patterns

### Index Organization

**Single Index**
- All documents in one vector space
- Use metadata filtering
- Best for: < 100K documents

**Multi-Index**
- Separate indices per category
- Faster search within category
- Best for: Distinct document types

**Hierarchical Index**
- Coarse-to-fine search
- First: Find relevant sections
- Then: Search within sections

### Embedding Strategies

**Dense Embeddings**
- TF-IDF (simple, no external deps)
- Sentence transformers (better quality)
- Domain-specific models

**Sparse Embeddings**
- BM25 for keyword matching
- Complement dense retrieval
- Good for exact phrase matches

## Context Window Management

### Context Assembly

**Top-K Retrieval**
```python
# Get top 5 most relevant chunks
results = vector_db.search(query, top_k=5)

# Assemble context
context = "\n\n".join([r['text'] for r in results])
```

**Score Thresholding**
```python
# Only include high-confidence results
results = [r for r in results if r['score'] > 0.7]
```

**Deduplication**
```python
# Remove duplicate or near-duplicate chunks
seen = set()
unique_results = []
for r in results:
    if r['id'] not in seen:
        unique_results.append(r)
        seen.add(r['id'])
```

### Token Budget Management

- Allocate context budget (e.g., 2000 tokens)
- Reserve space for query and response
- Truncate or summarize if needed
- Prioritize by relevance score

## Memory and Conversation Patterns

### Short-Term Memory

Store recent exchanges:
```python
memory = ConversationMemory(max_history=10)
memory.add_exchange(user_msg, assistant_msg)

# Retrieve relevant past exchanges
relevant = memory.search_history(current_query, top_k=3)
```

### Long-Term Memory

- Store summaries of conversations
- Extract key facts and preferences
- Update knowledge base incrementally

### Context Threading

Maintain conversation context:
1. Keep last N exchanges in memory
2. Search for relevant past discussions
3. Combine with document retrieval
4. Build comprehensive context

## RAG System Types

### 1. Document Q&A
- Single document or document set
- Question answering focus
- Direct fact extraction

### 2. Multi-Document RAG
- Cross-document synthesis
- Compare information sources
- Handle contradictions

### 3. Conversational RAG
- Multi-turn dialogue support
- Context from conversation + documents
- Reference resolution

### 4. Agentic RAG
- Query planning and routing
- Multi-step retrieval
- Tool usage integration

## Error Handling Patterns

### No Results Found

```python
if not results or max_score < threshold:
    # Fallback strategies:
    # 1. Broaden search (reduce filters)
    # 2. Query reformulation
    # 3. Return "no information" response
```

### Conflicting Information

```python
# Present multiple perspectives
# Include source attribution
# Let user decide or aggregate
```

### Poor Retrieval Quality

```python
# Signs:
# - Low similarity scores
# - User feedback indicates wrong context
# 
# Solutions:
# - Improve chunking strategy
# - Enhance query processing
# - Add more documents
# - Use better embeddings
```

## Performance Optimization

### Indexing Optimization

- **Batch embedding**: Process documents in batches
- **Incremental updates**: Add new docs without full reindex
- **Lazy loading**: Load embeddings on demand

### Search Optimization

- **Caching**: Cache common queries
- **Approximate search**: Trade accuracy for speed
- **Parallel retrieval**: Search multiple indices simultaneously

### Memory Optimization

- **Sparse storage**: Use sparse matrices when appropriate
- **Quantization**: Reduce embedding precision
- **Pruning**: Remove low-quality documents

## Testing Strategies

### Retrieval Quality Tests

```python
test_cases = [
    {
        "query": "What is machine learning?",
        "expected_docs": ["doc_1", "doc_5"],
        "min_score": 0.7
    }
]

for test in test_cases:
    results = rag_system.search(test["query"])
    assert any(r['id'] in test['expected_docs'] for r in results)
    assert max(r['score'] for r in results) > test['min_score']
```

### End-to-End Tests

- Query diversity (simple, complex, ambiguous)
- Edge cases (empty query, very long query)
- Multilingual queries (if applicable)
- Performance benchmarks

## Common Pitfalls

1. **Chunk size too large**: Loses precision, wastes tokens
2. **No overlap**: Misses information at boundaries
3. **Poor metadata**: Can't filter or route effectively
4. **Single retrieval strategy**: Misses some relevant docs
5. **No reranking**: Surface-level similarity only
6. **Ignoring context**: Each query treated independently
7. **No source attribution**: Can't verify or trace answers

## Integration Patterns

### With Knowledge Graphs

- Extract entities from retrieved chunks
- Query graph for relationships
- Enrich context with structured data

### With Traditional Search

- Use RAG for semantic search
- Fall back to keyword search
- Combine results with score fusion

### With External APIs

- Retrieve from local docs first
- Use APIs for missing information
- Cache API results in vector DB

## Evaluation Metrics

### Retrieval Metrics

- **Recall@K**: Relevant docs in top K results
- **Precision@K**: Proportion of relevant docs in top K
- **MRR**: Mean Reciprocal Rank of first relevant doc
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Quality

- Faithfulness: Answer supported by context
- Relevance: Answer addresses query
- Coherence: Well-formed response
- Completeness: All aspects covered

### User Metrics

- User satisfaction ratings
- Task completion rate
- Time to find information
- Return visit rate
