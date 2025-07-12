# Knowledge Graph Builder MCP Server

A Knowledge Graph Builder that transforms text or web content into structured knowledge graphs using local AI models with MCP (Model Context Protocol) integration for persistent storage in Neo4j and Qdrant.

## 🚀 Features

- **Local AI Processing**: Uses local models via Ollama or LM Studio for entity extraction
- **Large Content Support**: Handles arbitrarily large content (300MB+) via intelligent chunking
- **Web Content Extraction**: Scrapes and analyzes full web pages without size limits
- **Knowledge Graph Generation**: Creates structured graphs with entities and relationships
- **Smart Chunking**: Automatically chunks large content with sentence boundary detection
- **Entity Merging**: Intelligently merges duplicate entities across chunks
- **Real-Time Visualization**: Live SVG graph updates as chunks are processed
- **Interactive SVG Output**: Color-coded entity types with progress tracking
- **MCP Integration**: Stores data in Neo4j (graph database) and Qdrant (vector database)
- **UUID Tracking**: Generates UUIDv8 for unified entity tracking across systems
- **Gradio Interface**: User-friendly web interface with dual JSON/SVG output

## 📊 Entity Types Extracted

- **👥 PERSON**: Names, individuals, key figures
- **🏢 ORGANIZATION**: Companies, institutions, groups
- **📍 LOCATION**: Places, countries, regions, addresses
- **💡 CONCEPT**: Ideas, technologies, abstract concepts
- **📅 EVENT**: Specific events, occurrences, incidents
- **🔧 OTHER**: Miscellaneous entities not fitting other categories

## 🔧 Setup

### Requirements

```bash
pip install -r requirements.txt

# For full visualization capabilities:
pip install networkx matplotlib
```

### Environment Variables

For detailed configuration instructions and complete environment variables reference, see the [Configuration](#🎛️-configuration) section below.

**Quick Start Configuration:**
```bash
# Basic setup (uses sensible defaults)
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:latest

# Optional: Custom endpoints and processing limits
export OLLAMA_BASE_URL=http://localhost:11434
export CHUNK_SIZE=2000
export MAX_CHUNKS=0
```

**Note:** All environment variables are optional and have sensible defaults. The application will run without any configuration.

### Local Model Setup

**For Ollama:**
```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull a model
ollama pull llama3.2:latest
```

**For LM Studio:**
1. Download and install LM Studio
2. Load a model in the local server
3. Start the local server on port 1234

## 🏃 Running the Application

```bash
python app.py
```

The application will launch a Gradio interface with MCP server capabilities enabled.

## 📝 Usage

### Text Input
Paste any text content to analyze:
```
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California.
```

### URL Input
Provide a web URL to extract and analyze:
```
https://en.wikipedia.org/wiki/Artificial_intelligence
```

### Large Content Processing (300MB+ Files)

For very large content like LLM conversation extracts:

```bash
# Example: Processing a 300MB conversation log
# The system will automatically:
# 1. Detect large content (>2000 chars by default)
# 2. Split into intelligent chunks at sentence boundaries
# 3. Process each chunk with the local AI model
# 4. Merge and deduplicate entities/relationships
# 5. Store with full lineage tracking in hKG

# Processing will show progress:
# "Processing large content (314,572,800 chars) in chunks..."
# "Processing 157,286 chunks..."
# "Processing chunk 1/157,286 (2000 chars)..."
# "Merged results: 45,231 entities, 128,904 relationships"
```

### Output Format

The system returns a structured JSON knowledge graph:

```json
{
  "source": {
    "type": "text|url",
    "value": "input_value",
    "content_preview": "first 200 characters..."
  },
  "knowledge_graph": {
    "entities": [
      {
        "name": "Apple Inc.",
        "type": "ORGANIZATION",
        "description": "Technology company founded in 1976"
      }
    ],
    "relationships": [
      {
        "source": "Steve Jobs",
        "target": "Apple Inc.",
        "relationship": "FOUNDED",
        "description": "Steve Jobs founded Apple Inc."
      }
    ],
    "entity_count": 5,
    "relationship_count": 4
  },
  "visualization": {
    "svg_content": "<svg>...</svg>",
    "svg_file_path": "/path/to/knowledge_graph_12345678.svg",
    "visualization_available": true,
    "real_time_updates": false,
    "incremental_files_saved": 0,
    "entity_color_mapping": {
      "ORGANIZATION": "#4ECDC4",
      "PERSON": "#FF6B6B"
    },
    "svg_generation_timestamp": "2024-01-15T10:30:05Z",
    "visualization_engine": "networkx+matplotlib"
  },
  "metadata": {
    "model": "ollama:llama3.2:latest",
    "content_length": 150,
    "uuid": "xxxxxxxx-xxxx-8xxx-xxxx-xxxxxxxxxxxx",
    "neo4j_stored": true,
    "qdrant_stored": true,
    "timestamp": "2024-01-15T10:30:00Z",
    "hkg_metadata": {
      "processing_method": "single",
      "chunk_count": 1,
      "chunk_size": 2000,
      "chunk_overlap": 200,
      "source_type": "text",
      "supports_large_content": true,
      "max_content_size": "unlimited",
      "visualization_integration": {
        "real_time_visualization": false,
        "svg_files_generated": 1,
        "entity_color_tracking": true,
        "visualization_lineage": true,
        "incremental_updates": false,
        "neo4j_viz_metadata": true,
        "qdrant_viz_metadata": true
      }
    }
  }
}
```

## 🎨 Real-Time Graph Visualization

### SVG Generation Features
- **Color-Coded Entity Types**: Each entity type has a distinct color (Person=Red, Organization=Teal, Location=Blue, Concept=Green, Event=Yellow, Other=Plum)
- **Interactive Layout**: Automatic graph layout using NetworkX spring layout algorithm
- **Relationship Labels**: Edge labels showing relationship types between entities
- **Entity Information**: Node labels with entity names and types
- **Legend**: Automatic legend generation based on entity types present
- **Statistics**: Real-time entity and relationship counts

### Real-Time Processing for Large Content
- **Progress Tracking**: Visual progress bar showing chunk processing completion
- **Incremental Updates**: Graph updates after each chunk is processed
- **Live Statistics**: Running totals of entities and relationships discovered
- **Incremental File Saves**: Each chunk creates a timestamped SVG file
- **Final Visualization**: Complete graph saved as final SVG

### File Output
- **Single Content**: `knowledge_graph_<uuid8>.svg`
- **Large Content (Chunked)**: 
  - Incremental: `knowledge_graph_<uuid8>_chunk_0001.svg`, `chunk_0002.svg`, etc.
  - Final: `knowledge_graph_<uuid8>.svg`

### Example Large Content Processing
```bash
# Processing a 300MB conversation log:
# "Processing large content (314,572,800 chars) in chunks..."
# "Processing 157,286 chunks..."
# 
# Real-time updates:
# "Processing chunk 1/157,286 (2000 chars)..."
# "Real-time graph updated: Updated graph: 5 entities, 3 relationships (Chunk 1/157,286)"
# "Saved incremental graph: knowledge_graph_12345678_chunk_0001.svg"
# 
# "Processing chunk 2/157,286 (2000 chars)..."
# "Real-time graph updated: Updated graph: 12 entities, 8 relationships (Chunk 2/157,286)"
# "Saved incremental graph: knowledge_graph_12345678_chunk_0002.svg"
# 
# ... continues for all chunks ...
# 
# "Final results: 45,231 entities, 128,904 relationships"
# "Final SVG visualization saved: knowledge_graph_12345678.svg"
```

## 🗄️ hKG (Hybrid Knowledge Graph) Storage with Visualization Integration

### Neo4j Integration (Graph Database)
- Stores entities as nodes with properties and enhanced metadata
- Creates relationships between entities with lineage tracking
- Maintains UUIDv8 for entity tracking across all databases
- Tracks chunking metadata for large content processing
- Records processing method (single vs chunked)
- **NEW**: Visualization metadata in entity observations including:
  - SVG file paths and availability status
  - Entity color mappings for graph visualization
  - Real-time update tracking for chunked processing
  - Incremental file counts for large content processing
- Accessible via MCP server tools

### Qdrant Integration (Vector Database)  
- Stores knowledge graphs as vector embeddings with enhanced metadata
- Enables semantic search across graphs of any size
- Maintains metadata for each knowledge graph including chunk information
- Tracks content length, processing method, and chunk count
- Supports similarity search across large document collections
- **NEW**: Visualization lineage tracking including:
  - Entity type and color mapping information
  - SVG generation timestamps and file paths
  - Real-time visualization update history
  - Incremental SVG file tracking for large content
- Accessible via MCP server tools

### hKG Unified Tracking with Visualization Lineage
- **UUIDv8 Across All Systems**: Common ancestry-encoded identifiers
- **Content Lineage**: Track how large content was processed and chunked
- **Processing Metadata**: Record chunk size, overlap, and processing method
- **Entity Provenance**: Track which chunks contributed to each entity
- **Relationship Mapping**: Maintain relationships across chunk boundaries
- **Semantic Coherence**: Ensure knowledge graph consistency across databases
- **NEW - Visualization Lineage**: Complete tracking of visual representation:
  - **SVG File Provenance**: Track all generated visualization files
  - **Color Mapping Consistency**: Maintain entity color assignments across chunks
  - **Real-Time Update History**: Log all incremental visualization updates
  - **Cross-Database Visual Metadata**: Synchronized visualization tracking in both Neo4j and Qdrant
  - **Incremental Visualization Tracking**: Complete audit trail of real-time graph updates

## 🔧 Architecture

### Core Components

- **`app.py`**: Main application file with Gradio interface
- **`extract_text_from_url()`**: Web scraping functionality (app.py:41)
- **`chunk_text()`**: Smart content chunking with sentence boundary detection (app.py:214)
- **`merge_extraction_results()`**: Intelligent merging of chunk results (app.py:250)
- **`get_entity_color()`**: Entity type color mapping (app.py:299)
- **`create_knowledge_graph_svg()`**: SVG graph generation (app.py:311)
- **`RealTimeGraphVisualizer`**: Real-time incremental visualization (app.py:453)
- **`extract_entities_and_relationships()`**: AI-powered entity extraction with real-time updates (app.py:645)
- **`extract_entities_and_relationships_single()`**: Single chunk processing (app.py:722)
- **`build_knowledge_graph()`**: Main orchestration function with visualization (app.py:795)
- **`generate_uuidv8()`**: UUID generation for entity tracking (app.py:68)

### Data Flow with hKG Integration and Real-Time Visualization

1. **Input Processing**: Text or URL input validation
2. **Content Extraction**: Web scraping for URLs, direct text for text input
3. **Real-Time Visualizer Setup**: Initialize incremental graph visualization system
4. **Content Chunking**: Smart chunking for large content (>2000 chars) with sentence boundary detection
5. **AI Analysis with Live Updates**: Local model processes each chunk for entities/relationships
6. **Incremental Visualization**: Real-time SVG graph updates after each chunk completion
7. **Result Merging**: Intelligent deduplication and merging of entities/relationships across chunks
8. **hKG Metadata Creation**: Generate processing metadata for lineage tracking
9. **Graph Generation**: Structured knowledge graph creation with enhanced metadata
10. **Final Visualization**: Generate complete SVG graph with all entities and relationships
11. **hKG Storage**: Persistence in Neo4j (graph) and Qdrant (vector) with unified UUIDv8 tracking
12. **Output**: JSON response with complete knowledge graph, hKG metadata, and SVG visualization

## 🎛️ Configuration

### Environment Variables Reference

All configuration is handled through environment variables. The application provides sensible defaults for all settings, allowing it to run without any configuration while still offering full customization.

#### Complete Environment Variables Table

| Variable | Type | Default | Required | Description | Example Values |
|----------|------|---------|----------|-------------|----------------|
| `MODEL_PROVIDER` | string | `"ollama"` | No | AI model provider to use | `"ollama"`, `"lmstudio"` |
| `LOCAL_MODEL` | string | `"llama3.2:latest"` | No | Local model identifier | `"llama3.2:latest"`, `"mistral:7b"`, `"codellama:13b"` |
| `OLLAMA_BASE_URL` | string | `"http://localhost:11434"` | No | Ollama API endpoint | `"http://localhost:11434"`, `"http://192.168.1.100:11434"` |
| `LMSTUDIO_BASE_URL` | string | `"http://localhost:1234"` | No | LM Studio API endpoint | `"http://localhost:1234"`, `"http://127.0.0.1:1234"` |
| `CHUNK_SIZE` | integer | `2000` | No | Characters per chunk for AI processing | `1000`, `2000`, `4000`, `8000` |
| `CHUNK_OVERLAP` | integer | `200` | No | Overlap between chunks for context | `100`, `200`, `400`, `500` |
| `MAX_CHUNKS` | integer | `0` | No | Maximum chunks to process (0=unlimited) | `0`, `100`, `1000`, `5000` |
| `HF_TOKEN` | string | `None` | No | HuggingFace API token (legacy, unused) | `"hf_xxxxxxxxxxxx"` |

### Configuration Methods

#### 1. Environment Variables (Recommended)
```bash
# Core Model Configuration
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:latest
export OLLAMA_BASE_URL=http://localhost:11434

# Large Content Processing
export CHUNK_SIZE=2000
export CHUNK_OVERLAP=200
export MAX_CHUNKS=0
```

#### 2. Shell Configuration (.bashrc/.zshrc)
```bash
# Add to ~/.bashrc or ~/.zshrc
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:latest
export OLLAMA_BASE_URL=http://localhost:11434
export CHUNK_SIZE=2000
export CHUNK_OVERLAP=200
export MAX_CHUNKS=0
```

#### 3. Python Environment File (.env)
```bash
# Create .env file in project root
MODEL_PROVIDER=ollama
LOCAL_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
MAX_CHUNKS=0
```

### Model Provider Configuration

#### Ollama Configuration (Default)
```bash
# Basic Ollama setup
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:latest
export OLLAMA_BASE_URL=http://localhost:11434

# Alternative models
export LOCAL_MODEL=mistral:7b          # Mistral 7B
export LOCAL_MODEL=codellama:13b       # Code Llama 13B
export LOCAL_MODEL=llama3.2:3b         # Llama 3.2 3B (faster)
export LOCAL_MODEL=phi3:mini           # Phi-3 Mini (lightweight)

# Remote Ollama instance
export OLLAMA_BASE_URL=http://192.168.1.100:11434
```

#### LM Studio Configuration
```bash
# Basic LM Studio setup
export MODEL_PROVIDER=lmstudio
export LOCAL_MODEL=any-model-name      # Model name is flexible for LM Studio
export LMSTUDIO_BASE_URL=http://localhost:1234

# Custom LM Studio port
export LMSTUDIO_BASE_URL=http://localhost:8080

# Remote LM Studio instance
export LMSTUDIO_BASE_URL=http://192.168.1.200:1234
```

### Large Content Processing Configuration

#### Chunk Size Optimization
```bash
# Small chunks (faster processing, more chunks)
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=100

# Medium chunks (balanced performance)
export CHUNK_SIZE=2000    # Default
export CHUNK_OVERLAP=200  # Default

# Large chunks (fewer chunks, more context)
export CHUNK_SIZE=4000
export CHUNK_OVERLAP=400

# Very large chunks (maximum context, slower)
export CHUNK_SIZE=8000
export CHUNK_OVERLAP=800
```

#### Processing Limits
```bash
# Unlimited processing (default)
export MAX_CHUNKS=0

# Process only first 100 chunks (testing)
export MAX_CHUNKS=100

# Process first 1000 chunks (moderate datasets)
export MAX_CHUNKS=1000

# Process first 10000 chunks (large datasets)
export MAX_CHUNKS=10000
```

### Performance Tuning Guidelines

#### For Speed Optimization
```bash
# Smaller chunks, less overlap, limited processing
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=50
export MAX_CHUNKS=500
export LOCAL_MODEL=llama3.2:3b  # Faster model
```

#### For Quality Optimization
```bash
# Larger chunks, more overlap, unlimited processing
export CHUNK_SIZE=4000
export CHUNK_OVERLAP=400
export MAX_CHUNKS=0
export LOCAL_MODEL=llama3.2:latest  # Full model
```

#### For Memory-Constrained Systems
```bash
# Balanced settings for limited resources
export CHUNK_SIZE=1500
export CHUNK_OVERLAP=150
export MAX_CHUNKS=1000
export LOCAL_MODEL=phi3:mini  # Lightweight model
```

### Configuration Validation

The application performs automatic validation of configuration settings:

- **Model Provider**: Validates `MODEL_PROVIDER` is either `"ollama"` or `"lmstudio"`
- **URLs**: Validates that provider URLs are accessible
- **Numeric Values**: Ensures `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MAX_CHUNKS` are valid integers
- **Model Availability**: Checks if the specified model is available on the provider

### Configuration Troubleshooting

#### Common Issues and Solutions

**1. Model Provider Not Responding**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check if LM Studio is running
curl http://localhost:1234/v1/models

# Solution: Start the appropriate service
ollama serve  # For Ollama
# Or start LM Studio GUI and enable local server
```

**2. Model Not Found**
```bash
# List available Ollama models
ollama list

# Pull missing model
ollama pull llama3.2:latest

# For LM Studio: Load model in GUI
```

**3. Memory Issues with Large Content**
```bash
# Reduce chunk size and set limits
export CHUNK_SIZE=1000
export MAX_CHUNKS=100

# Use lighter model
export LOCAL_MODEL=llama3.2:3b
```

**4. Slow Processing**
```bash
# Optimize for speed
export CHUNK_SIZE=1500
export CHUNK_OVERLAP=100
export MAX_CHUNKS=500
export LOCAL_MODEL=phi3:mini
```

### Example Configuration Scenarios

#### Scenario 1: Development Setup
```bash
# Fast iteration, limited processing
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:3b
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=100
export MAX_CHUNKS=50
```

#### Scenario 2: Production Setup
```bash
# High quality, unlimited processing
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:latest
export CHUNK_SIZE=3000
export CHUNK_OVERLAP=300
export MAX_CHUNKS=0
```

#### Scenario 3: Large Dataset Processing
```bash
# Optimized for 300MB+ files
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=llama3.2:latest
export CHUNK_SIZE=2000
export CHUNK_OVERLAP=200
export MAX_CHUNKS=0
```

#### Scenario 4: Resource-Constrained Environment
```bash
# Minimal resource usage
export MODEL_PROVIDER=ollama
export LOCAL_MODEL=phi3:mini
export CHUNK_SIZE=800
export CHUNK_OVERLAP=50
export MAX_CHUNKS=200
```

### Advanced Configuration

#### Custom Model Endpoints
```bash
# Docker-based Ollama
export OLLAMA_BASE_URL=http://ollama-container:11434

# Kubernetes service
export OLLAMA_BASE_URL=http://ollama-service.default.svc.cluster.local:11434

# Load balancer
export OLLAMA_BASE_URL=http://ollama-lb.example.com:11434
```

#### Dynamic Configuration
The application reads environment variables at startup. To change configuration:

1. Set new environment variables
2. Restart the application
3. Configuration changes take effect immediately

### Error Handling

Comprehensive error handling for:
- Invalid URLs or network failures
- Missing local models or API endpoints
- JSON parsing errors from LLM responses
- Malformed or empty inputs
- Database connection issues
- Invalid configuration values
- Model provider connectivity issues
- Memory constraints during large content processing

## 🔍 hKG MCP Integration with Visual Lineage

The application integrates with MCP servers for hybrid knowledge graph storage with complete visualization tracking:
- **Neo4j**: Graph database storage and querying with enhanced metadata + visualization lineage
- **Qdrant**: Vector database for semantic search with chunk tracking + visual metadata
- **Unified Tracking**: UUIDv8 across all storage systems for entity lineage + visualization provenance
- **Metadata Persistence**: Processing method, chunk count, content lineage + SVG generation tracking
- **Large Content Support**: Seamless handling of 300MB+ content via chunking + real-time visualization
- **Visualization Integration**: Complete visual representation tracking across all storage systems

### Enhanced hKG Features via MCP
- **Entity Provenance**: Track which content chunks contributed to each entity + their visual representation
- **Relationship Lineage**: Maintain relationships across chunk boundaries + visual edge tracking
- **Content Ancestry**: UUIDv8 encoding for hierarchical content tracking + visualization file lineage
- **Processing Audit**: Complete record of how large content was processed + visualization generation
- **Semantic Search**: Vector similarity across knowledge graphs of any size + visual metadata search
- **NEW - Visual Lineage**: Complete visualization tracking including:
  - **SVG File Provenance**: Track all generated visualization files with timestamps
  - **Entity Color Consistency**: Maintain color mappings across all chunks and storage systems
  - **Real-Time Visualization History**: Log every incremental graph update during processing
  - **Cross-Database Visual Sync**: Synchronized visualization metadata in Neo4j and Qdrant
  - **Incremental Visualization Audit**: Complete trail of real-time updates for large content

### Visualization-Enhanced Storage
- **Neo4j Entity Observations** now include:
  - SVG file paths and generation status
  - Entity color assignments for visual consistency
  - Real-time update counts for chunked processing
  - Visualization availability and engine information
- **Qdrant Vector Content** now includes:
  - Entity color mapping information for similarity search
  - SVG generation timestamps and file paths
  - Real-time visualization update metadata
  - Incremental file tracking for large content visualization

MCP tools are automatically available when running in Claude Code environment with MCP servers configured.

## 🎯 hKG Visualization Architecture

### Integrated Visualization Lineage System
The hKG system now maintains complete visualization lineage alongside traditional knowledge graph storage:

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Source Text   │───▶│  Chunking + AI       │───▶│  Entity/Relation    │
│   (300MB+)      │    │  Processing          │    │  Extraction         │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                 │                           │
                                 ▼                           ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ Real-Time SVG   │◀───│  Incremental Graph   │◀───│  Merged Results     │
│ Generation      │    │  Visualization       │    │  + Deduplication    │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
         │                        │                           │
         ▼                        ▼                           ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ SVG File        │    │  Visualization       │    │  hKG Storage        │
│ Storage         │    │  Metadata Creation   │    │  (Neo4j + Qdrant)  │
│ (Incremental)   │    │                      │    │  + Viz Metadata     │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Visualization Metadata Flow
1. **Real-Time Updates**: Each chunk generates incremental SVG with progress tracking
2. **Color Consistency**: Entity colors maintained across all chunks and storage systems
3. **File Lineage**: Complete audit trail of all generated SVG files
4. **Cross-Database Sync**: Visualization metadata synchronized in both Neo4j and Qdrant
5. **Provenance Tracking**: Link between source chunks, entities, and their visual representation

### hKG Benefits for Large Content (300MB+)
- **Visual Progress Monitoring**: Real-time graph evolution during processing
- **Chunk-Level Visualization**: Individual SVG files for each processing stage
- **Complete Audit Trail**: Full lineage from source text to final visualization
- **Cross-Reference Capability**: Link entities back to their source chunks and visual appearance
- **Scalable Visualization**: Handles arbitrarily large graphs with consistent performance

## 📊 Development

### Project Structure
```
KGB-mcp/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── CLAUDE.md             # Claude Code instructions
├── ARCHITECTURE.md       # System architecture
├── test_core.py          # Core functionality tests
└── test_integration.py   # Integration tests
```

### Testing
```bash
# Run core tests
python test_core.py

# Run integration tests
python test_integration.py
```

Transform any content into structured knowledge graphs with the power of local AI and MCP integration!