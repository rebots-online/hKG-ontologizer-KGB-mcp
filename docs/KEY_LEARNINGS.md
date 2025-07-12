# Key Learnings - hKG Ontologizer KGB MCP

This document captures critical learnings from the development and troubleshooting of the Knowledge Graph Builder MCP project, designed to benefit future projects and prevent rediscovering the same solutions.

## 🔄 AsyncIO Event Loop Management in UI Applications

### Problem
**Error**: `"asyncio.run() cannot be called from a running event loop"`

**Context**: Gradio applications with async backend functions

### Root Cause
- Gradio runs in its own event loop
- `asyncio.run()` creates a new event loop and cannot be called from within an existing one
- Mixing sync UI callbacks with async backend functions creates conflict

### Solution Pattern
```python
def sync_wrapper_for_gradio(input_data):
    """Robust pattern for Gradio async integration"""
    import asyncio
    import concurrent.futures
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Use thread pool for isolation
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(async_function(input_data))
                )
                return future.result()
        else:
            return loop.run_until_complete(async_function(input_data))
    except RuntimeError:
        return asyncio.run(async_function(input_data))
```

### Key Principles
1. **Never use `asyncio.run()` inside async functions** - use `await` instead
2. **Detect event loop state** before creating new loops
3. **Use ThreadPoolExecutor** for sync/async boundary isolation
4. **Test in target environment** - behavior differs between standalone and framework contexts

### Cross-Project Value
- **High** - Affects any Python application mixing async/await with UI frameworks
- **Technologies**: Gradio, Streamlit, FastAPI + UI, Jupyter, MCP servers
- **Pattern**: Reusable wrapper pattern for similar integrations

---

## 🏗️ hKG (Hybrid Knowledge Graph) Architecture Insights

### UUIDv8 for Cross-System Entity Tracking
- **Purpose**: Unified entity identification across Neo4j, Qdrant, and PostgreSQL
- **Implementation**: Timestamp-based UUIDv8 generation with namespace encoding
- **Benefit**: Enables cross-database correlation and lineage tracking

### Real-Time Visualization with Incremental Updates
- **Pattern**: Progressive graph building with SVG generation at each chunk
- **Files**: Incremental SVG saves for large content processing
- **Storage**: Visualization metadata tracked in both Neo4j and Qdrant

### Chunked Processing for Large Content
- **Threshold**: 2000 character chunks with 200 character overlap
- **Benefits**: Handles 300MB+ content, real-time progress, memory efficiency
- **Challenge**: Maintaining entity consistency across chunks

---

## 🔧 Development Process Learnings

### MCP Integration Strategy
- **Local Development**: Mock MCP responses for rapid iteration
- **Production**: Full MCP server integration with Claude Code environment
- **Testing**: Dual-mode operation (standalone + MCP-enabled)

### Error Handling Philosophy
- **Graceful Degradation**: Application functions even without optional components
- **Verbose Logging**: Detailed progress tracking for debugging
- **User Communication**: Clear error messages with actionable guidance

### Configuration Management
- **Environment Variables**: All external service URLs and keys
- **Provider Flexibility**: Support for Ollama, LM Studio, and hosted APIs
- **Model Discovery**: Dynamic model listing from active providers

---

## 📊 Performance and Scalability Insights

### Memory Management
- **Streaming Processing**: Chunk-based handling prevents memory overflow
- **SVG Generation**: Temporary file cleanup to prevent disk bloat
- **Vector Storage**: Embedding generation with fallback strategies

### Concurrency Patterns
- **Async/Await**: Native async for I/O-bound operations
- **Thread Pools**: Isolation for event loop conflicts
- **Progress Callbacks**: Real-time updates without blocking

---

## 🔗 Cross-Project Reusable Patterns

### 1. Sync/Async UI Bridge Pattern
**Location**: `app.py:1796-1831`
**Reusability**: High - any Python UI framework with async backend

### 2. UUIDv8 Entity Tracking
**Location**: `app.py:255-271`
**Reusability**: High - any multi-database system requiring correlation

### 3. Chunked Content Processing
**Location**: `app.py:513-547`
**Reusability**: High - any large content processing system

### 4. Real-Time Visualization Updates
**Location**: `app.py:737-928`
**Reusability**: Medium - graph visualization with progress tracking

### 5. Multi-Provider Model Integration
**Location**: `app.py:273-327`
**Reusability**: High - any AI application supporting multiple providers

---

## 🎯 Prevention Guidelines for Future Projects

### AsyncIO Integration
1. Design clear sync/async boundaries from project start
2. Use event loop detection in all wrapper functions
3. Test with both standalone and framework execution
4. Document async/sync interfaces clearly

### Knowledge Graph Projects
1. Plan for large content from the beginning
2. Implement incremental processing early
3. Design for multiple storage backends
4. Include visualization in the core architecture

### MCP Server Development
1. Build with dual-mode operation (local + MCP)
2. Mock external services for development
3. Implement comprehensive error handling
4. Plan for various deployment environments

---

## 📚 Documentation Strategy

### What Worked
- **Inline Code Comments**: Explain complex async patterns
- **Troubleshooting Guides**: Step-by-step problem resolution
- **Architecture Diagrams**: Visual representation of data flow
- **Key Learnings**: Cross-project knowledge transfer

### What to Improve
- **Performance Benchmarks**: Document throughput and memory usage
- **Deployment Guides**: Various environment setup instructions
- **Integration Examples**: More third-party service examples

---

## 🔄 Continuous Learning Integration

This document is connected to the hKG system with the following entities:
- **Technical Learning**: AsyncIO event loop management (UUID: a6bf77b5-b73d-463d-ae5d-832fcd7920f4)
- **Best Practice**: Event loop management patterns (UUID: 5b1a3443-7eb9-4483-8258-10a7b369a3cc)
- **Project Learning**: Problem resolution process (UUID: a34a1067-ac19-4e54-90fc-67569f95b9ff)

These learnings are searchable and referenceable by future projects through the hKG system, ensuring knowledge transfer and preventing rediscovery of solutions.

---

*Last Updated: 2025-07-12*
*Project: hKG-ontologizer-KGB-mcp*
*Learning Integration: hKG System via MCP*