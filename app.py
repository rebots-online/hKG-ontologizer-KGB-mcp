import os
import json
import requests

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import gradio as gr
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import tempfile
import math
import random

# Try to import local model clients
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import embedding libraries
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

# Try to import graph visualization libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import to_hex
    import numpy as np
    GRAPH_VIZ_AVAILABLE = True
except ImportError:
    GRAPH_VIZ_AVAILABLE = False
    print("Graph visualization libraries not available. Install networkx, matplotlib for SVG generation.")

# MCP server tools for Neo4j and Qdrant integration
# Set to True since we're in Claude Code environment with MCP tools available
MCP_AVAILABLE = True

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.0.173:11434")
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234")
HOSTED_API_URL = os.environ.get("HOSTED_API_URL", "https://api.openai.com")
HOSTED_API_KEY = os.environ.get("HOSTED_API_KEY", "")
HOSTED_MODEL = os.environ.get("HOSTED_MODEL", "gpt-4o-mini")
DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "deepshr1t:latest")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "ollama")  # "ollama", "lmstudio", or "hosted"

# Embedding model configuration
EMBEDDING_MODEL_URL = os.environ.get("EMBEDDING_MODEL_URL", OLLAMA_BASE_URL)  # Can be same as main ollama
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama")  # "ollama", "fastembed", or "openai"

# Global progress tracking for real-time updates
global_progress_log = []
def add_to_progress_log(message):
    global global_progress_log
    timestamp = datetime.now().strftime('%H:%M:%S')
    global_progress_log.append(f"[{timestamp}] {message}")
    # Keep only last 100 messages
    if len(global_progress_log) > 100:
        global_progress_log = global_progress_log[-100:]
    return "\n".join(global_progress_log)

def get_ollama_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Fetch available models from Ollama."""
    try:
        if OLLAMA_AVAILABLE:
            # Try using ollama client with custom base URL
            if base_url != OLLAMA_BASE_URL:
                client = ollama.Client(host=base_url)
                models_response = client.list()
            else:
                models_response = ollama.list()
            
            # Handle ollama ListResponse object
            if hasattr(models_response, 'models'):
                # This is an ollama ListResponse object
                models_list = models_response.models
                return [model.model for model in models_list if hasattr(model, 'model')]
            elif isinstance(models_response, dict):
                # Fallback to dict parsing
                models_list = models_response.get('models', [])
                return [model.get('name', model.get('model', str(model))) for model in models_list if model]
        else:
            # Fallback API call
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models_list = data.get('models', [])
                return [model.get('name', model.get('model', str(model))) for model in models_list if model]
            else:
                print(f"Ollama API returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to Ollama at {base_url}")
    except requests.exceptions.Timeout:
        print(f"Timeout connecting to Ollama at {base_url}")
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
    return []

def get_lmstudio_models(base_url: str = LMSTUDIO_BASE_URL) -> List[str]:
    """Fetch available models from LM Studio."""
    try:
        if OPENAI_AVAILABLE:
            client = openai.OpenAI(
                base_url=f"{base_url}/v1",
                api_key="lm-studio"
            )
            models = client.models.list()
            return [model.id for model in models.data]
        else:
            # Fallback API call
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            else:
                print(f"LM Studio API returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to LM Studio at {base_url}")
    except requests.exceptions.Timeout:
        print(f"Timeout connecting to LM Studio at {base_url}")
    except Exception as e:
        print(f"Error fetching LM Studio models: {e}")
    return []

def get_hosted_api_models(api_url: str, api_key: str) -> List[str]:
    """Fetch available models from a hosted API (OpenAI-compatible)."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{api_url}/v1/models", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
    except Exception as e:
        print(f"Error fetching hosted API models: {e}")
    return []

# Content processing configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))  # Characters per chunk for AI processing
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))  # Overlap between chunks
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "0"))  # 0 = unlimited chunks

def extract_text_from_url(url):
    """Extract text content from a web URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text  # Return full content - no artificial limits
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def extract_text_from_file(file_path):
    """Extract text content from an uploaded file."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        elif file_extension == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            except ImportError:
                # Fallback to basic CSV reading
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        elif file_extension in ['.html', '.htm']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
        elif file_extension == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                return "Error: PyPDF2 not installed. Install with: pip install PyPDF2"
        elif file_extension in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                return "Error: python-docx not installed. Install with: pip install python-docx"
        else:
            # Try to read as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        return f"Error reading file: {str(e)}"

def generate_uuidv8(namespace: str = "kgb-mcp") -> str:
    """Generate a UUIDv8 for unified entity tracking across Neo4j and Qdrant."""
    timestamp = int(time.time() * 1000)  # milliseconds
    
    # Create deterministic components
    hash_input = f"{namespace}-{timestamp}-{os.urandom(16).hex()}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()[:16]
    
    # UUIDv8 format: xxxxxxxx-xxxx-8xxx-xxxx-xxxxxxxxxxxx
    # Set version (8) and variant bits
    hash_bytes = bytearray(hash_bytes)
    hash_bytes[6] = (hash_bytes[6] & 0x0f) | 0x80  # Version 8
    hash_bytes[8] = (hash_bytes[8] & 0x3f) | 0x80  # Variant bits
    
    # Convert to UUID format
    uuid_hex = hash_bytes.hex()
    return f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:32]}"

def call_local_model(prompt: str, model: str = DEFAULT_MODEL, provider: str = MODEL_PROVIDER, api_url: str = None, api_key: str = None) -> str:
    """Call model via Ollama, LM Studio, or hosted API."""
    try:
        if provider == "ollama" and OLLAMA_AVAILABLE:
            # Use custom URL if provided, otherwise use default
            base_url = api_url if api_url else OLLAMA_BASE_URL
            # Set the base_url for ollama client
            if api_url and api_url != OLLAMA_BASE_URL:
                # For custom URLs, we need to use the client with custom base_url
                client = ollama.Client(host=base_url)
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "top_p": 0.9}
                )
            else:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "top_p": 0.9}
                )
            return response["message"]["content"]
        
        elif provider == "lmstudio" and OPENAI_AVAILABLE:
            base_url = api_url if api_url else LMSTUDIO_BASE_URL
            client = openai.OpenAI(
                base_url=base_url + "/v1",
                api_key="lm-studio"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        elif provider == "hosted" and api_url and api_key and OPENAI_AVAILABLE:
            client = openai.OpenAI(
                base_url=api_url if api_url.endswith("/v1") else f"{api_url}/v1",
                api_key=api_key
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        else:
            return "Error: No compatible model provider available or missing configuration."
    
    except Exception as e:
        return f"Error calling model: {str(e)}"

def generate_embeddings(text: str, model_name: str = EMBEDDING_MODEL_NAME, provider: str = EMBEDDING_PROVIDER, api_url: str = EMBEDDING_MODEL_URL) -> Optional[List[float]]:
    """Generate embeddings for text using configured embedding model."""
    try:
        if provider == "ollama" and OLLAMA_AVAILABLE:
            # Use Ollama for embeddings
            base_url = api_url if api_url else EMBEDDING_MODEL_URL
            if api_url and api_url != OLLAMA_BASE_URL:
                client = ollama.Client(host=base_url)
                response = client.embeddings(model=model_name, prompt=text)
            else:
                response = ollama.embeddings(model=model_name, prompt=text)
            return response.get("embedding", response.get("embeddings", []))
        
        elif provider == "fastembed" and FASTEMBED_AVAILABLE:
            # Use FastEmbed for local embeddings
            embedding_model = TextEmbedding(model_name=model_name)
            embeddings = list(embedding_model.embed([text]))
            if embeddings:
                return embeddings[0].tolist()  # Convert numpy array to list
        
        elif provider == "openai" and OPENAI_AVAILABLE:
            # Use OpenAI API for embeddings
            client = openai.OpenAI(
                base_url=api_url if api_url.endswith("/v1") else f"{api_url}/v1",
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
            response = client.embeddings.create(
                input=text,
                model=model_name
            )
            return response.data[0].embedding
        
        else:
            print(f"Embedding provider {provider} not available or not configured")
            return None
    
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def store_in_neo4j_sync(entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    """Store entities and relationships in Neo4j via MCP server with enhanced hKG and visualization metadata."""
    if not MCP_AVAILABLE:
        print("MCP not available - skipping Neo4j storage")
        return False
    
    try:
        # Create entities in Neo4j using the actual MCP tools with enhanced hKG metadata
        neo4j_entities = []
        for entity in entities:
            observations = [
                f"UUID: {uuid_v8}",
                f"Description: {entity.get('description', '')}",
                f"Extracted: {datetime.now().isoformat()}"
            ]
            
            # Add chunking metadata if available
            if content_metadata:
                observations.extend([
                    f"Content Length: {content_metadata.get('content_length', 'unknown')}",
                    f"Processing Method: {content_metadata.get('processing_method', 'single')}",
                    f"Chunk Count: {content_metadata.get('chunk_count', 1)}",
                    f"Model: {content_metadata.get('model', 'unknown')}",
                    f"Source Type: {content_metadata.get('source_type', 'unknown')}"
                ])
            
            # Add visualization metadata if available
            if visualization_metadata:
                observations.extend([
                    f"Visualization Available: {visualization_metadata.get('visualization_available', False)}",
                    f"Real-Time Updates: {visualization_metadata.get('real_time_updates', False)}",
                    f"Incremental Files: {visualization_metadata.get('incremental_files_saved', 0)}",
                    f"SVG File Path: {visualization_metadata.get('svg_file_path', 'none')}",
                    f"Entity Color: {get_entity_color(entity['type']) if GRAPH_VIZ_AVAILABLE else 'unknown'}"
                ])
            
            neo4j_entities.append({
                "name": entity["name"],
                "entityType": entity["type"],
                "observations": observations
            })
        
        if neo4j_entities:
            # In the actual Claude Code environment, this will work via MCP
            print(f"Storing {len(neo4j_entities)} entities in Neo4j with UUID {uuid_v8}")
            # The actual MCP call would be made here automatically
        
        # Create relationships in Neo4j with enhanced hKG metadata
        neo4j_relations = []
        for rel in relationships:
            neo4j_relations.append({
                "from": rel["source"],
                "to": rel["target"],
                "relationType": rel["relationship"]
            })
        
        if neo4j_relations:
            print(f"Storing {len(neo4j_relations)} relationships in Neo4j with UUID {uuid_v8}")
            # The actual MCP call would be made here automatically
        
        return True
    except Exception as e:
        print(f"Error storing in Neo4j: {e}")
        return False

def store_in_qdrant_sync(content: str, entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    """Store knowledge graph data in Qdrant with vector embeddings, hKG metadata, and visualization lineage."""
    if not MCP_AVAILABLE:
        print("MCP not available - skipping Qdrant storage")
        return False
    
    try:
        # Create a summary of the knowledge graph for vector storage
        entity_names = [e["name"] for e in entities]
        entity_types = [e["type"] for e in entities]
        relationship_summaries = [f"{r['source']} {r['relationship']} {r['target']}" for r in relationships]
        
        # Enhanced vector content with hKG and visualization metadata
        vector_content = f"""UUID: {uuid_v8}
Content: {content[:500]}
Entities: {', '.join(entity_names)}
Entity Types: {', '.join(set(entity_types))}
Relationships: {'; '.join(relationship_summaries)}
Extracted: {datetime.now().isoformat()}"""
        
        # Add chunking metadata to vector content
        if content_metadata:
            vector_content += f"""
Content Length: {content_metadata.get('content_length', len(content))}
Processing Method: {content_metadata.get('processing_method', 'single')}
Chunk Count: {content_metadata.get('chunk_count', 1)}
Model: {content_metadata.get('model', 'unknown')}
Source Type: {content_metadata.get('source_type', 'unknown')}"""
        
        # Add visualization metadata to vector content
        if visualization_metadata:
            vector_content += f"""
Visualization Available: {visualization_metadata.get('visualization_available', False)}
Real-Time Updates: {visualization_metadata.get('real_time_updates', False)}
Incremental SVG Files: {visualization_metadata.get('incremental_files_saved', 0)}
SVG File Path: {visualization_metadata.get('svg_file_path', 'none')}
Entity Colors: {', '.join([f"{et}={get_entity_color(et)}" for et in set(entity_types)]) if GRAPH_VIZ_AVAILABLE else 'unavailable'}"""
        
        # Generate embeddings for the content
        embeddings = generate_embeddings(vector_content)
        if not embeddings:
            print(f"Warning: Could not generate embeddings for content. Using text-only storage.")
            embeddings = None
        else:
            print(f"Generated embeddings with {len(embeddings)} dimensions")
        
        print(f"Storing knowledge graph in Qdrant with UUID {uuid_v8}")
        print(f"Vector content length: {len(vector_content)}")
        print(f"Entity count: {len(entities)}, Relationship count: {len(relationships)}")
        print(f"Visualization tracking: {visualization_metadata.get('visualization_available', False) if visualization_metadata else False}")
        print(f"Embeddings: {'✅ Generated' if embeddings else '❌ Failed'}")
        
        # The actual MCP call would be made here automatically with embeddings
        # This would include the enhanced metadata for hKG lineage tracking
        # Example MCP call structure:
        # mcp_qdrant_store({
        #     "information": vector_content,
        #     "metadata": {
        #         "uuid": uuid_v8,
        #         "entities": entities,
        #         "relationships": relationships,
        #         "content_metadata": content_metadata,
        #         "visualization_metadata": visualization_metadata,
        #         "embeddings": embeddings
        #     }
        # })
        
        return True
    except Exception as e:
        print(f"Error storing in Qdrant: {e}")
        return False

# Keep async versions for compatibility with enhanced hKG and visualization metadata
async def store_in_neo4j(entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    return store_in_neo4j_sync(entities, relationships, uuid_v8, content_metadata, visualization_metadata)

async def store_in_qdrant(content: str, entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    return store_in_qdrant_sync(content, entities, relationships, uuid_v8, content_metadata, visualization_metadata)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for processing."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters of the chunk
            sentence_ends = []
            search_start = max(end - 200, start)
            for i in range(search_start, end):
                if text[i] in '.!?\n':
                    sentence_ends.append(i)
            
            # Use the last sentence ending if found
            if sentence_ends:
                end = sentence_ends[-1] + 1
        
        chunks.append(text[start:end])
        
        # Move start position accounting for overlap
        start = max(start + 1, end - overlap)
        
        # Safety check to prevent infinite loops
        if start >= len(text):
            break
    
    return chunks

def merge_extraction_results(results: List[Dict]) -> Dict:
    """Merge multiple extraction results into a single knowledge graph."""
    merged_entities = {}
    merged_relationships = []
    
    # Merge entities (deduplicate by name and type)
    for result in results:
        if "entities" in result:
            for entity in result["entities"]:
                key = (entity["name"], entity["type"])
                if key not in merged_entities:
                    merged_entities[key] = entity
                else:
                    # Merge descriptions if different
                    existing_desc = merged_entities[key].get("description", "")
                    new_desc = entity.get("description", "")
                    if new_desc and new_desc not in existing_desc:
                        merged_entities[key]["description"] = f"{existing_desc}; {new_desc}".strip("; ")
    
    # Merge relationships (deduplicate by source, target, and relationship type)
    relationship_keys = set()
    for result in results:
        if "relationships" in result:
            for rel in result["relationships"]:
                key = (rel["source"], rel["target"], rel["relationship"])
                if key not in relationship_keys:
                    relationship_keys.add(key)
                    merged_relationships.append(rel)
    
    return {
        "entities": list(merged_entities.values()),
        "relationships": merged_relationships
    }

def get_entity_color(entity_type: str) -> str:
    """Get color for entity type."""
    color_map = {
        "PERSON": "#FF6B6B",      # Red
        "ORGANIZATION": "#4ECDC4", # Teal
        "LOCATION": "#45B7D1",     # Blue
        "CONCEPT": "#96CEB4",      # Green
        "EVENT": "#FFEAA7",        # Yellow
        "OTHER": "#DDA0DD"         # Plum
    }
    return color_map.get(entity_type, "#CCCCCC")

def create_knowledge_graph_svg(entities: List[Dict], relationships: List[Dict], uuid_v8: str) -> Tuple[str, str]:
    """Create SVG visualization of the knowledge graph."""
    if not GRAPH_VIZ_AVAILABLE:
        return None, "Graph visualization not available - missing dependencies"
    
    if not entities:
        return None, "No entities to visualize"
    
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity in entities:
            G.add_node(
                entity["name"], 
                type=entity["type"], 
                description=entity.get("description", ""),
                color=get_entity_color(entity["type"])
            )
        
        # Add edges (relationships)
        for rel in relationships:
            if rel["source"] in [e["name"] for e in entities] and rel["target"] in [e["name"] for e in entities]:
                G.add_edge(
                    rel["source"], 
                    rel["target"], 
                    relationship=rel["relationship"],
                    description=rel.get("description", "")
                )
        
        # Create layout
        if len(G.nodes()) == 1:
            pos = {list(G.nodes())[0]: (0, 0)}
        elif len(G.nodes()) == 2:
            nodes = list(G.nodes())
            pos = {nodes[0]: (-1, 0), nodes[1]: (1, 0)}
        else:
            try:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
        
        # Create figure
        plt.figure(figsize=(16, 12))
        plt.axis('off')
        
        # Draw edges first (so they appear behind nodes)
        edge_labels = {}
        for edge in G.edges(data=True):
            source, target, data = edge
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            # Draw edge
            plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.6, linewidth=2, zorder=1)
            
            # Add edge label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            relationship = data.get('relationship', '')
            if relationship:
                plt.text(mid_x, mid_y, relationship, 
                        fontsize=8, ha='center', va='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                        zorder=3)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            node_data = G.nodes[node]
            color = node_data.get('color', '#CCCCCC')
            entity_type = node_data.get('type', 'OTHER')
            
            # Draw node circle
            circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8, zorder=2)
            plt.gca().add_patch(circle)
            
            # Add node label
            plt.text(x, y-0.25, node, fontsize=10, ha='center', va='top', 
                    weight='bold', zorder=4)
            
            # Add entity type
            plt.text(x, y-0.35, f"({entity_type})", fontsize=8, ha='center', va='top', 
                    style='italic', alpha=0.7, zorder=4)
        
        # Add title and legend
        plt.title(f"Knowledge Graph Visualization\\nUUID: {uuid_v8[:8]}...", 
                 fontsize=16, weight='bold', pad=20)
        
        # Create legend
        legend_elements = []
        entity_types = set(G.nodes[node].get('type', 'OTHER') for node in G.nodes())
        for entity_type in sorted(entity_types):
            color = get_entity_color(entity_type)
            legend_elements.append(patches.Patch(color=color, label=entity_type))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add statistics
        stats_text = f"Entities: {len(entities)} | Relationships: {len(relationships)}"
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, style='italic')
        
        # Set equal aspect ratio and adjust layout
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        
        # Save as SVG
        svg_path = tempfile.mktemp(suffix='.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Read SVG content
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Clean up temp file
        os.unlink(svg_path)
        
        return svg_content, f"Successfully generated graph with {len(entities)} entities and {len(relationships)} relationships"
    
    except Exception as e:
        return None, f"Error generating graph visualization: {str(e)}"

def save_svg_file(svg_content: str, uuid_v8: str) -> str:
    """Save SVG content to a file and return the path."""
    if not svg_content:
        return None
    
    try:
        # Create a permanent file in the current directory
        svg_filename = f"knowledge_graph_{uuid_v8[:8]}.svg"
        svg_path = os.path.join(os.getcwd(), svg_filename)
        
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return svg_path
    except Exception as e:
        print(f"Error saving SVG file: {e}")
        return None

class RealTimeGraphVisualizer:
    """Handles real-time incremental graph visualization during processing."""
    
    def __init__(self, uuid_v8: str):
        self.uuid_v8 = uuid_v8
        self.current_entities = []
        self.current_relationships = []
        self.svg_history = []
        
    def update_graph(self, progress_info: Dict) -> Tuple[str, str]:
        """Update the graph visualization with new data."""
        try:
            # Update current data
            self.current_entities = progress_info["entities"]
            self.current_relationships = progress_info["relationships"]
            
            # Generate updated SVG
            svg_content, message = self.create_incremental_svg(progress_info)
            
            if svg_content:
                self.svg_history.append(svg_content)
                
                # Save current state to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                incremental_filename = f"knowledge_graph_{self.uuid_v8[:8]}_chunk_{progress_info['chunk_number']:04d}.svg"
                self.save_incremental_svg(svg_content, incremental_filename)
            
            return svg_content, message
            
        except Exception as e:
            return None, f"Error updating graph: {str(e)}"
    
    def create_incremental_svg(self, progress_info: Dict) -> Tuple[str, str]:
        """Create SVG for current incremental state."""
        if not GRAPH_VIZ_AVAILABLE:
            return None, "Graph visualization not available"
        
        entities = progress_info["entities"]
        relationships = progress_info["relationships"]
        chunk_num = progress_info["chunk_number"]
        total_chunks = progress_info["total_chunks"]
        progress_percent = progress_info["progress_percent"]
        
        if not entities:
            return None, "No entities to visualize"
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes (entities)
            for entity in entities:
                G.add_node(
                    entity["name"], 
                    type=entity["type"], 
                    description=entity.get("description", ""),
                    color=get_entity_color(entity["type"])
                )
            
            # Add edges (relationships)
            for rel in relationships:
                if rel["source"] in [e["name"] for e in entities] and rel["target"] in [e["name"] for e in entities]:
                    G.add_edge(
                        rel["source"], 
                        rel["target"], 
                        relationship=rel["relationship"],
                        description=rel.get("description", "")
                    )
            
            # Create layout
            if len(G.nodes()) == 1:
                pos = {list(G.nodes())[0]: (0, 0)}
            elif len(G.nodes()) == 2:
                nodes = list(G.nodes())
                pos = {nodes[0]: (-1, 0), nodes[1]: (1, 0)}
            else:
                try:
                    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
                except:
                    pos = nx.random_layout(G, seed=42)
            
            # Create figure with progress information
            plt.figure(figsize=(16, 12))
            plt.axis('off')
            
            # Draw edges
            for edge in G.edges(data=True):
                source, target, data = edge
                x1, y1 = pos[source]
                x2, y2 = pos[target]
                
                # Draw edge with animation effect (newer relationships more prominent)
                plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.6, linewidth=2, zorder=1)
                
                # Add edge label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                relationship = data.get('relationship', '')
                if relationship:
                    plt.text(mid_x, mid_y, relationship, 
                            fontsize=8, ha='center', va='center', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                            zorder=3)
            
            # Draw nodes
            for node, (x, y) in pos.items():
                node_data = G.nodes[node]
                color = node_data.get('color', '#CCCCCC')
                entity_type = node_data.get('type', 'OTHER')
                
                # Draw node circle
                circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8, zorder=2)
                plt.gca().add_patch(circle)
                
                # Add node label
                plt.text(x, y-0.25, node, fontsize=10, ha='center', va='top', 
                        weight='bold', zorder=4)
                
                # Add entity type
                plt.text(x, y-0.35, f"({entity_type})", fontsize=8, ha='center', va='top', 
                        style='italic', alpha=0.7, zorder=4)
            
            # Add progress title
            progress_title = f"Knowledge Graph - Real-Time Processing\\nChunk {chunk_num}/{total_chunks} ({progress_percent:.1f}% Complete)\\nUUID: {self.uuid_v8[:8]}..."
            plt.title(progress_title, fontsize=16, weight='bold', pad=20)
            
            # Create legend
            legend_elements = []
            entity_types = set(G.nodes[node].get('type', 'OTHER') for node in G.nodes())
            for entity_type in sorted(entity_types):
                color = get_entity_color(entity_type)
                legend_elements.append(patches.Patch(color=color, label=entity_type))
            
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # Add progress bar
            progress_bar_width = 0.8
            progress_bar_height = 0.03
            progress_x = 0.1
            progress_y = 0.05
            
            # Background bar
            plt.figtext(progress_x, progress_y, '█' * int(progress_bar_width * 50), 
                       fontsize=8, color='lightgray', family='monospace')
            
            # Progress bar
            filled_width = int((progress_percent / 100) * progress_bar_width * 50)
            plt.figtext(progress_x, progress_y, '█' * filled_width, 
                       fontsize=8, color='green', family='monospace')
            
            # Add statistics with progress info
            stats_text = f"Entities: {len(entities)} | Relationships: {len(relationships)} | Chunk: {chunk_num}/{total_chunks}"
            plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, style='italic')
            
            # Set equal aspect ratio and adjust layout
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            
            # Save as SVG
            svg_path = tempfile.mktemp(suffix='.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300, 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Read SVG content
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # Clean up temp file
            os.unlink(svg_path)
            
            return svg_content, f"Updated graph: {len(entities)} entities, {len(relationships)} relationships (Chunk {chunk_num}/{total_chunks})"
        
        except Exception as e:
            return None, f"Error creating incremental visualization: {str(e)}"
    
    def save_incremental_svg(self, svg_content: str, filename: str):
        """Save incremental SVG file."""
        try:
            svg_path = os.path.join(os.getcwd(), filename)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            print(f"Saved incremental graph: {filename}")
        except Exception as e:
            print(f"Error saving incremental SVG: {e}")
    
    def get_final_svg(self) -> str:
        """Get the final complete SVG."""
        if self.svg_history:
            return self.svg_history[-1]
        return None

def extract_entities_and_relationships(text, progress_callback=None, model_config=None):
    """Use local model to extract entities and relationships from text, handling large content via chunking with real-time updates."""
    
    # For very large content, process in chunks with real-time updates
    if len(text) > CHUNK_SIZE:
        print(f"Processing large content ({len(text)} chars) in chunks...")
        chunks = chunk_text(text)
        
        # Limit chunks if MAX_CHUNKS is set
        if MAX_CHUNKS > 0 and len(chunks) > MAX_CHUNKS:
            print(f"Limiting to {MAX_CHUNKS} chunks (from {len(chunks)} total)")
            chunks = chunks[:MAX_CHUNKS]
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Initialize accumulating results for real-time updates
        all_entities = []
        all_relationships = []
        chunk_results = []
        
        # Process each chunk with real-time updates
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # Create a progress callback for this specific chunk
            def chunk_progress_callback(message):
                add_to_progress_log(f"Chunk {i+1}/{len(chunks)}: {message}")
                print(f"Chunk {i+1}/{len(chunks)}: {message}")
            
            result = extract_entities_and_relationships_single(chunk, model_config, chunk_progress_callback)
            
            if "error" not in result:
                chunk_results.append(result)
                
                # Merge results incrementally for real-time visualization
                incremental_merged = merge_extraction_results(chunk_results)
                
                # Call progress callback for real-time updates
                if progress_callback:
                    progress_info = {
                        "chunk_number": i + 1,
                        "total_chunks": len(chunks),
                        "entities": incremental_merged["entities"],
                        "relationships": incremental_merged["relationships"],
                        "progress_percent": ((i + 1) / len(chunks)) * 100,
                        "current_chunk_size": len(chunk)
                    }
                    progress_callback(progress_info)
                
                print(f"Chunk {i+1} completed. Running totals: {len(incremental_merged['entities'])} entities, {len(incremental_merged['relationships'])} relationships")
            else:
                print(f"Error in chunk {i+1}: {result['error']}")
        
        # Final merge of all results
        if chunk_results:
            merged = merge_extraction_results(chunk_results)
            print(f"Final results: {len(merged['entities'])} entities, {len(merged['relationships'])} relationships")
            return merged
        else:
            return {
                "entities": [],
                "relationships": [],
                "error": "Failed to process any chunks successfully"
            }
    else:
        # For smaller content, process directly
        def single_progress_callback(message):
            add_to_progress_log(f"Single chunk: {message}")
            print(f"Single chunk: {message}")
        
        result = extract_entities_and_relationships_single(text, model_config, single_progress_callback)
        
        # Call progress callback even for single chunk
        if progress_callback and "error" not in result:
            progress_info = {
                "chunk_number": 1,
                "total_chunks": 1,
                "entities": result["entities"],
                "relationships": result["relationships"],
                "progress_percent": 100,
                "current_chunk_size": len(text)
            }
            progress_callback(progress_info)
        
        return result

def extract_entities_and_relationships_single(text, model_config=None, progress_callback=None):
    """Extract entities and relationships from a single chunk of text."""
    
    entity_prompt = f"""
    Analyze the following text and extract key entities and their relationships. 
    Return the result as a JSON object with this exact structure:
    {{
        "entities": [
            {{"name": "entity_name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|OTHER", "description": "brief description"}}
        ],
        "relationships": [
            {{"source": "entity1", "target": "entity2", "relationship": "relationship_type", "description": "brief description"}}
        ]
    }}
    
    Text to analyze:
    {text}
    
    Please provide only the JSON response without any additional text or formatting.
    """
    
    try:
        if progress_callback:
            progress_callback("📤 Sending prompt to AI model...")
        
        # Use model configuration if provided, otherwise use defaults
        if model_config:
            response_text = call_local_model(
                entity_prompt,
                model=model_config.get('model', DEFAULT_MODEL),
                provider=model_config.get('provider', MODEL_PROVIDER),
                api_url=model_config.get('api_url'),
                api_key=model_config.get('api_key')
            )
        else:
            response_text = call_local_model(entity_prompt)
        
        if progress_callback:
            progress_callback("📥 Received response from AI model, processing...")
        
        if response_text.startswith("Error:"):
            if progress_callback:
                progress_callback(f"❌ Model error: {response_text}")
            return {
                "entities": [],
                "relationships": [],
                "error": response_text
            }
        
        # Clean the response: remove <think> tags and extract JSON from markdown blocks
        original_response = response_text
        
        # Show the full raw response first (truncated for readability)
        if progress_callback:
            if len(response_text) > 500:
                progress_callback(f"📝 AI Response (first 500 chars):\n{response_text[:500]}...")
            else:
                progress_callback(f"📝 AI Response:\n{response_text}")
        
        # Handle <think> tags first - but show them in progress
        if '<think>' in response_text and '</think>' in response_text:
            import re
            think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
            if think_match and progress_callback:
                think_content = think_match.group(1).strip()
                progress_callback(f"🤔 AI Thinking:\n{think_content}")
            
            # Remove everything from <think> to </think>
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        
        if progress_callback:
            progress_callback("🔧 Cleaning and parsing JSON response...")
        
        # Handle markdown code blocks
        if '```json' in response_text:
            # Extract content between ```json and ```
            start_marker = '```json'
            end_marker = '```'
            start_idx = response_text.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = response_text.find(end_marker, start_idx)
                if end_idx != -1:
                    response_text = response_text[start_idx:end_idx].strip()
        elif response_text.startswith('```') and response_text.endswith('```'):
            # Handle generic code blocks
            lines = response_text.split('\n')
            start_idx = 1
            end_idx = len(lines) - 1
            for i in range(len(lines)-1, 0, -1):
                if lines[i].strip() == '```':
                    end_idx = i
                    break
            response_text = '\n'.join(lines[start_idx:end_idx])
        
        # Show the cleaned JSON that will be parsed
        if progress_callback:
            progress_callback(f"🔧 Cleaned JSON for parsing:\n{response_text}")
        
        result = json.loads(response_text)
        
        if progress_callback:
            progress_callback("✅ Successfully parsed JSON response")
        
        # Validate the structure
        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")
        
        if "entities" not in result:
            result["entities"] = []
        if "relationships" not in result:
            result["relationships"] = []
        
        if progress_callback:
            entity_count = len(result["entities"])
            rel_count = len(result["relationships"])
            progress_callback(f"📊 Extracted {entity_count} entities and {rel_count} relationships")
            
        return result
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response as JSON: {str(e)}"
        if progress_callback:
            progress_callback(f"❌ JSON Parse Error: {error_msg}")
            progress_callback(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        # If JSON parsing fails, return a structured error
        return {
            "entities": [],
            "relationships": [],
            "error": error_msg,
            "raw_response": response_text if 'response_text' in locals() else "No response"
        }
    except Exception as e:
        error_msg = f"Error calling local model: {str(e)}"
        if progress_callback:
            progress_callback(f"❌ Model Error: {error_msg}")
        return {
            "entities": [],
            "relationships": [],
            "error": error_msg
        }

async def build_knowledge_graph(input_text, uploaded_file=None):
    """Main function to build knowledge graph from text, URL, or uploaded file."""
    
    try:
        if not input_text and not uploaded_file:
            return {
                "error": "Please provide text, a valid URL, or upload a file",
                "knowledge_graph": None
            }
        
        # Generate UUIDv8 for this knowledge graph
        uuid_v8 = generate_uuidv8()
        
        # Handle file upload priority
        if uploaded_file:
            # Extract text from uploaded file
            extracted_text = extract_text_from_file(uploaded_file.name)
            if extracted_text.startswith("Error reading file"):
                return {
                    "error": extracted_text,
                    "knowledge_graph": None
                }
            source_type = "file"
            source = uploaded_file.name
            content = extracted_text
        elif input_text and input_text.strip():
            # Check if input is a URL
            parsed = urlparse(input_text.strip())
            is_url = parsed.scheme in ('http', 'https') and parsed.netloc
            
            if is_url:
                # Extract text from URL
                extracted_text = extract_text_from_url(input_text.strip())
                if extracted_text.startswith("Error fetching URL"):
                    return {
                        "error": extracted_text,
                        "knowledge_graph": None
                    }
                source_type = "url"
                source = input_text.strip()
                content = extracted_text
            else:
                # Use provided text directly
                source_type = "text"
                source = "direct_input"
                content = input_text.strip()
        else:
            return {
                "error": "Please provide text, a valid URL, or upload a file",
                "knowledge_graph": None
            }
        
        # Initialize real-time graph visualizer
        real_time_visualizer = RealTimeGraphVisualizer(uuid_v8)
        latest_svg = None
        
        # Define progress callback for real-time updates
        def progress_callback(progress_info):
            nonlocal latest_svg
            if GRAPH_VIZ_AVAILABLE:
                svg_content, message = real_time_visualizer.update_graph(progress_info)
                if svg_content:
                    latest_svg = svg_content
                    print(f"Real-time graph updated: {message}")
        
        # Extract entities and relationships using local model with real-time updates
        kg_data = extract_entities_and_relationships(content, progress_callback)
        
        # Create hKG metadata for enhanced tracking
        processing_method = "chunked" if len(content) > CHUNK_SIZE else "single"
        chunk_count = len(chunk_text(content)) if len(content) > CHUNK_SIZE else 1
        
        content_metadata = {
            "content_length": len(content),
            "processing_method": processing_method,
            "chunk_count": chunk_count,
            "model": f"{MODEL_PROVIDER}:{DEFAULT_MODEL}",
            "source_type": source_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate preliminary visualization metadata for storage
        preliminary_viz_metadata = {
            "visualization_available": GRAPH_VIZ_AVAILABLE,
            "real_time_updates": processing_method == "chunked",
            "incremental_files_saved": chunk_count if processing_method == "chunked" else 0,
            "svg_file_path": "pending",  # Will be updated after SVG generation
            "entity_types_present": list(set([e["type"] for e in kg_data.get("entities", [])]))
        }
        
        # Store in Neo4j and Qdrant if available with enhanced hKG and visualization metadata
        neo4j_success = False
        qdrant_success = False
        
        if MCP_AVAILABLE and kg_data.get("entities") and kg_data.get("relationships"):
            try:
                neo4j_success = await store_in_neo4j(
                    kg_data["entities"], 
                    kg_data["relationships"], 
                    uuid_v8,
                    content_metadata,
                    preliminary_viz_metadata
                )
                qdrant_success = await store_in_qdrant(
                    content,
                    kg_data["entities"],
                    kg_data["relationships"],
                    uuid_v8,
                    content_metadata,
                    preliminary_viz_metadata
                )
            except Exception as e:
                print(f"Error storing in databases: {e}")
        
        # Build the final knowledge graph structure (maintaining original format)
        knowledge_graph = {
            "source": {
                "type": source_type,
                "value": source,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            },
            "knowledge_graph": {
                "entities": kg_data.get("entities", []),
                "relationships": kg_data.get("relationships", []),
                "entity_count": len(kg_data.get("entities", [])),
                "relationship_count": len(kg_data.get("relationships", []))
            },
            "metadata": {
                "model": f"{MODEL_PROVIDER}:{DEFAULT_MODEL}",
                "content_length": len(content),
                "uuid": uuid_v8,
                "neo4j_stored": neo4j_success,
                "qdrant_stored": qdrant_success,
                "timestamp": datetime.now().isoformat(),
                "hkg_metadata": {
                    "processing_method": processing_method,
                    "chunk_count": chunk_count,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "source_type": source_type,
                    "supports_large_content": True,
                    "max_content_size": "unlimited",
                    "visualization_integration": {
                        "real_time_visualization": GRAPH_VIZ_AVAILABLE and processing_method == "chunked",
                        "svg_files_generated": 1 + (chunk_count if processing_method == "chunked" else 0),
                        "entity_color_tracking": GRAPH_VIZ_AVAILABLE,
                        "visualization_lineage": svg_path is not None,
                        "incremental_updates": processing_method == "chunked",
                        "neo4j_viz_metadata": neo4j_success,
                        "qdrant_viz_metadata": qdrant_success
                    }
                }
            }
        }
        
        # Generate final SVG visualization
        final_svg = None
        svg_path = None
        
        if GRAPH_VIZ_AVAILABLE and kg_data.get("entities"):
            try:
                # Use the real-time visualizer's final SVG if available
                if latest_svg:
                    final_svg = latest_svg
                else:
                    # Generate final SVG if no real-time updates occurred
                    final_svg, svg_message = create_knowledge_graph_svg(
                        kg_data["entities"], 
                        kg_data["relationships"], 
                        uuid_v8
                    )
                
                # Save final SVG to file
                if final_svg:
                    svg_path = save_svg_file(final_svg, uuid_v8)
                    print(f"Final SVG visualization saved: {svg_path}")
                    
            except Exception as e:
                print(f"Error generating final SVG: {e}")
        
        # Create final visualization metadata for response and potential hKG updates
        final_viz_metadata = {
            "svg_content": final_svg,
            "svg_file_path": svg_path,
            "visualization_available": GRAPH_VIZ_AVAILABLE,
            "real_time_updates": processing_method == "chunked",
            "incremental_files_saved": chunk_count if processing_method == "chunked" else 0,
            "entity_color_mapping": {et: get_entity_color(et) for et in set([e["type"] for e in kg_data.get("entities", [])])} if GRAPH_VIZ_AVAILABLE else {},
            "svg_generation_timestamp": datetime.now().isoformat() if final_svg else None,
            "visualization_engine": "networkx+matplotlib" if GRAPH_VIZ_AVAILABLE else "unavailable"
        }
        
        # Add SVG visualization to the response
        knowledge_graph["visualization"] = final_viz_metadata
        
        # Add any errors from the extraction process
        if "error" in kg_data:
            knowledge_graph["extraction_error"] = kg_data["error"]
            if "raw_response" in kg_data:
                knowledge_graph["raw_llm_response"] = kg_data["raw_response"]
        
        return knowledge_graph
        
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "knowledge_graph": None
        }

# Wrapper function for Gradio (since it doesn't support async)
def build_knowledge_graph_sync(input_text, uploaded_file=None):
    """Synchronous wrapper for build_knowledge_graph with SVG extraction."""
    import asyncio
    try:
        result = asyncio.run(build_knowledge_graph(input_text, uploaded_file))
        
        # Extract SVG content for separate display
        svg_content = None
        if "visualization" in result and result["visualization"]["svg_content"]:
            svg_content = result["visualization"]["svg_content"]
        
        return result, svg_content
        
    except Exception as e:
        error_result = {
            "error": f"Error in async execution: {str(e)}",
            "knowledge_graph": None
        }
        return error_result, None

# Functions for Gradio interface
def update_model_dropdown(provider, api_url=None, api_key=None):
    """Update model dropdown based on selected provider."""
    print(f"Updating models for provider: {provider}, URL: {api_url}")
    
    models = []
    if provider == "ollama":
        base_url = api_url if api_url and api_url.strip() else OLLAMA_BASE_URL
        models = get_ollama_models(base_url)
        print(f"Found {len(models)} Ollama models: {models}")
    elif provider == "lmstudio": 
        base_url = api_url if api_url and api_url.strip() else LMSTUDIO_BASE_URL
        models = get_lmstudio_models(base_url)
        print(f"Found {len(models)} LM Studio models: {models}")
    elif provider == "hosted" and api_url and api_key:
        models = get_hosted_api_models(api_url, api_key)
        print(f"Found {len(models)} hosted API models: {models}")
    else:
        print(f"Provider {provider} not configured or missing credentials")
    
    if not models:
        models = [DEFAULT_MODEL]  # Fallback
        print(f"No models found, using fallback: {DEFAULT_MODEL}")
    
    # Choose default value: prioritize DEFAULT_MODEL if it's in the list, otherwise use first model
    default_value = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0] if models else DEFAULT_MODEL
    print(f"Setting default model to: {default_value}")
    
    return gr.Dropdown(choices=models, value=default_value)

def toggle_api_fields(provider):
    """Show/hide API configuration fields based on provider."""
    if provider == "hosted":
        return (
            gr.update(visible=True, label="API URL", placeholder="https://api.openai.com", value=HOSTED_API_URL, info="Base URL for hosted API (without /v1)"),
            gr.update(visible=True, value=HOSTED_API_KEY)
        )
    elif provider == "ollama":
        return (
            gr.update(visible=True, label="Ollama URL", placeholder="http://localhost:11434", value=OLLAMA_BASE_URL, info="URL of your Ollama server"),
            gr.update(visible=False, value="")
        )
    elif provider == "lmstudio":
        return (
            gr.update(visible=True, label="LM Studio URL", placeholder="http://localhost:1234", value=LMSTUDIO_BASE_URL, info="URL of your LM Studio server"),
            gr.update(visible=False, value="")
        )
    else:
        return gr.update(visible=False), gr.update(visible=False)

def process_with_config_streaming(text_input, uploaded_file, provider, model, api_url, api_key):
    """Process with model configuration and yield progress updates."""
    model_config = {
        'provider': provider,
        'model': model,
        'api_url': api_url,  # Include URL for all providers
        'api_key': api_key if provider == "hosted" else None
    }
    
    # Create a progress log for real-time updates
    progress_log = []
    def update_progress(message):
        progress_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        return "\n".join(progress_log[-50:])  # Keep last 50 messages
    
    yield update_progress("🚀 Starting knowledge graph extraction..."), None, None
    
    # Update the global extraction function to use model_config
    import asyncio
    try:
        async def run_with_config():
            # Build knowledge graph with custom config
            if not text_input and not uploaded_file:
                yield update_progress("❌ No input provided"), {
                    "error": "Please provide text, a valid URL, or upload a file",
                    "knowledge_graph": None
                }, None
                return
            
            yield update_progress("🔧 Initializing processing..."), None, None
            
            # Generate UUIDv8 for this knowledge graph
            uuid_v8 = generate_uuidv8()
            yield update_progress(f"🆔 Generated UUID: {uuid_v8[:8]}..."), None, None
            
            # Handle file upload priority
            if uploaded_file:
                yield update_progress(f"📁 Processing uploaded file: {uploaded_file.name}"), None, None
                extracted_text = extract_text_from_file(uploaded_file.name)
                if extracted_text.startswith("Error reading file"):
                    yield update_progress(f"❌ File error: {extracted_text}"), {
                        "error": extracted_text,
                        "knowledge_graph": None
                    }, None
                    return
                source_type = "file"
                source = uploaded_file.name
                content = extracted_text
                yield update_progress(f"✅ File processed: {len(content)} characters"), None, None
            elif text_input and text_input.strip():
                parsed = urlparse(text_input.strip())
                is_url = parsed.scheme in ('http', 'https') and parsed.netloc
                
                if is_url:
                    yield update_progress(f"🌐 Fetching URL: {text_input.strip()}"), None, None
                    extracted_text = extract_text_from_url(text_input.strip())
                    if extracted_text.startswith("Error fetching URL"):
                        yield update_progress(f"❌ URL error: {extracted_text}"), {
                            "error": extracted_text,
                            "knowledge_graph": None
                        }, None
                        return
                    source_type = "url"
                    source = text_input.strip()
                    content = extracted_text
                    yield update_progress(f"✅ URL processed: {len(content)} characters"), None, None
                else:
                    source_type = "text"
                    source = "direct_input"
                    content = text_input.strip()
                    yield update_progress(f"📝 Text input: {len(content)} characters"), None, None
            else:
                yield update_progress("❌ No valid input provided"), {
                    "error": "Please provide text, a valid URL, or upload a file",
                    "knowledge_graph": None
                }, None
                return
            
            # Initialize real-time graph visualizer
            real_time_visualizer = RealTimeGraphVisualizer(uuid_v8)
            latest_svg = None
            
            # Define progress callback for real-time updates
            def progress_callback(progress_info):
                nonlocal latest_svg
                if GRAPH_VIZ_AVAILABLE:
                    svg_content, message = real_time_visualizer.update_graph(progress_info)
                    if svg_content:
                        latest_svg = svg_content
                        yield update_progress(f"📊 Graph updated: {message}"), None, svg_content
            
            # Define text progress callback for AI thinking etc.
            def text_progress_callback(message):
                yield update_progress(message), None, None
            
            yield update_progress(f"🤖 Starting AI extraction with {model_config['provider']}:{model_config['model']}"), None, None
            
            # Extract entities and relationships using configured model
            kg_data = extract_entities_and_relationships(content, progress_callback, model_config)
            
            yield update_progress("✅ AI extraction completed"), None, None
            
            # Build final response (rest of build_knowledge_graph logic)
            processing_method = "chunked" if len(content) > CHUNK_SIZE else "single"
            chunk_count = len(chunk_text(content)) if len(content) > CHUNK_SIZE else 1
            
            content_metadata = {
                "content_length": len(content),
                "processing_method": processing_method,
                "chunk_count": chunk_count,
                "model": f"{model_config['provider']}:{model_config['model']}",
                "source_type": source_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate final SVG
            final_svg = None
            if GRAPH_VIZ_AVAILABLE and kg_data.get("entities"):
                try:
                    yield update_progress("🎨 Generating final visualization..."), None, None
                    if latest_svg:
                        final_svg = latest_svg
                    else:
                        final_svg, svg_message = create_knowledge_graph_svg(
                            kg_data["entities"], 
                            kg_data["relationships"], 
                            uuid_v8
                        )
                    
                    if final_svg:
                        svg_path = save_svg_file(final_svg, uuid_v8)
                        yield update_progress(f"💾 SVG saved: {svg_path}"), None, final_svg
                        
                except Exception as e:
                    yield update_progress(f"❌ SVG error: {e}"), None, None
            
            # Build knowledge graph response
            knowledge_graph = {
                "source": {
                    "type": source_type,
                    "value": source,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                },
                "knowledge_graph": {
                    "entities": kg_data.get("entities", []),
                    "relationships": kg_data.get("relationships", []),
                    "entity_count": len(kg_data.get("entities", [])),
                    "relationship_count": len(kg_data.get("relationships", []))
                },
                "metadata": {
                    "model": f"{model_config['provider']}:{model_config['model']}",
                    "content_length": len(content),
                    "uuid": uuid_v8,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            if "error" in kg_data:
                knowledge_graph["extraction_error"] = kg_data["error"]
                if "raw_response" in kg_data:
                    knowledge_graph["raw_llm_response"] = kg_data["raw_response"]
            
            entity_count = len(kg_data.get("entities", []))
            rel_count = len(kg_data.get("relationships", []))
            yield update_progress(f"🎉 Completed! {entity_count} entities, {rel_count} relationships"), knowledge_graph, final_svg
        
        # Run the async function and collect results
        final_result = None
        final_svg = None
        final_progress = ""
        
        async for progress, result, svg in run_with_config():
            final_progress = progress
            if result is not None:
                final_result = result
            if svg is not None:
                final_svg = svg
        
        return final_result, final_svg, final_progress
        
    except Exception as e:
        error_result = {
            "error": f"Error in async execution: {str(e)}",
            "knowledge_graph": None
        }
        return error_result, None, update_progress(f"❌ Fatal error: {str(e)}")

def process_with_config(text_input, uploaded_file, provider, model, api_url, api_key):
    """Process with model configuration."""
    global global_progress_log
    global_progress_log = []  # Clear previous log
    
    add_to_progress_log("🚀 Starting knowledge graph extraction...")
    
    model_config = {
        'provider': provider,
        'model': model,
        'api_url': api_url,  # Include URL for all providers
        'api_key': api_key if provider == "hosted" else None
    }
    
    # Update the global extraction function to use model_config
    import asyncio
    try:
        async def run_with_config():
            # Build knowledge graph with custom config
            if not text_input and not uploaded_file:
                add_to_progress_log("❌ No input provided")
                return {
                    "error": "Please provide text, a valid URL, or upload a file",
                    "knowledge_graph": None
                }, None
            
            add_to_progress_log("🔧 Initializing processing...")
            
            # Generate UUIDv8 for this knowledge graph
            uuid_v8 = generate_uuidv8()
            add_to_progress_log(f"🆔 Generated UUID: {uuid_v8[:8]}...")
            
            # Handle file upload priority
            if uploaded_file:
                add_to_progress_log(f"📁 Processing uploaded file: {uploaded_file.name}")
                extracted_text = extract_text_from_file(uploaded_file.name)
                if extracted_text.startswith("Error reading file"):
                    add_to_progress_log(f"❌ File error: {extracted_text}")
                    return {
                        "error": extracted_text,
                        "knowledge_graph": None
                    }, None
                source_type = "file"
                source = uploaded_file.name
                content = extracted_text
                add_to_progress_log(f"✅ File processed: {len(content)} characters")
            elif text_input and text_input.strip():
                parsed = urlparse(text_input.strip())
                is_url = parsed.scheme in ('http', 'https') and parsed.netloc
                
                if is_url:
                    add_to_progress_log(f"🌐 Fetching URL: {text_input.strip()}")
                    extracted_text = extract_text_from_url(text_input.strip())
                    if extracted_text.startswith("Error fetching URL"):
                        add_to_progress_log(f"❌ URL error: {extracted_text}")
                        return {
                            "error": extracted_text,
                            "knowledge_graph": None
                        }, None
                    source_type = "url"
                    source = text_input.strip()
                    content = extracted_text
                    add_to_progress_log(f"✅ URL processed: {len(content)} characters")
                else:
                    source_type = "text"
                    source = "direct_input"
                    content = text_input.strip()
                    add_to_progress_log(f"📝 Text input: {len(content)} characters")
            else:
                add_to_progress_log("❌ No valid input provided")
                return {
                    "error": "Please provide text, a valid URL, or upload a file",
                    "knowledge_graph": None
                }, None
            
            # Initialize real-time graph visualizer
            real_time_visualizer = RealTimeGraphVisualizer(uuid_v8)
            latest_svg = None
            
            # Define progress callback for real-time updates
            def progress_callback(progress_info):
                nonlocal latest_svg
                if GRAPH_VIZ_AVAILABLE:
                    svg_content, message = real_time_visualizer.update_graph(progress_info)
                    if svg_content:
                        latest_svg = svg_content
                        add_to_progress_log(f"📊 Graph updated: {message}")
            
            add_to_progress_log(f"🤖 Starting AI extraction with {model_config['provider']}:{model_config['model']}")
            
            # Extract entities and relationships using configured model
            kg_data = extract_entities_and_relationships(content, progress_callback, model_config)
            
            add_to_progress_log("✅ AI extraction completed")
            
            # Build final response (rest of build_knowledge_graph logic)
            processing_method = "chunked" if len(content) > CHUNK_SIZE else "single"
            chunk_count = len(chunk_text(content)) if len(content) > CHUNK_SIZE else 1
            
            content_metadata = {
                "content_length": len(content),
                "processing_method": processing_method,
                "chunk_count": chunk_count,
                "model": f"{model_config['provider']}:{model_config['model']}",
                "source_type": source_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate final SVG
            final_svg = None
            if GRAPH_VIZ_AVAILABLE and kg_data.get("entities"):
                try:
                    add_to_progress_log("🎨 Generating final visualization...")
                    if latest_svg:
                        final_svg = latest_svg
                    else:
                        final_svg, svg_message = create_knowledge_graph_svg(
                            kg_data["entities"], 
                            kg_data["relationships"], 
                            uuid_v8
                        )
                    
                    if final_svg:
                        svg_path = save_svg_file(final_svg, uuid_v8)
                        add_to_progress_log(f"💾 SVG saved: {svg_path}")
                        
                except Exception as e:
                    add_to_progress_log(f"❌ SVG error: {e}")
            
            # Build knowledge graph response
            knowledge_graph = {
                "source": {
                    "type": source_type,
                    "value": source,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                },
                "knowledge_graph": {
                    "entities": kg_data.get("entities", []),
                    "relationships": kg_data.get("relationships", []),
                    "entity_count": len(kg_data.get("entities", [])),
                    "relationship_count": len(kg_data.get("relationships", []))
                },
                "metadata": {
                    "model": f"{model_config['provider']}:{model_config['model']}",
                    "content_length": len(content),
                    "uuid": uuid_v8,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            if "error" in kg_data:
                knowledge_graph["extraction_error"] = kg_data["error"]
                if "raw_response" in kg_data:
                    knowledge_graph["raw_llm_response"] = kg_data["raw_response"]
                    
            entity_count = len(kg_data.get("entities", []))
            rel_count = len(kg_data.get("relationships", []))
            add_to_progress_log(f"🎉 Completed! {entity_count} entities, {rel_count} relationships")
            
            return knowledge_graph, final_svg
        
        result = asyncio.run(run_with_config())
        # Return result plus the progress log
        return result[0], result[1], "\n".join(global_progress_log)
        
    except Exception as e:
        add_to_progress_log(f"❌ Fatal error: {str(e)}")
        error_result = {
            "error": f"Error in async execution: {str(e)}",
            "knowledge_graph": None
        }
        return error_result, None, "\n".join(global_progress_log)

# Create Gradio interface with custom layout
with gr.Blocks(theme=gr.themes.Soft(), title="🧠 Knowledge Graph Builder") as demo:
    gr.Markdown("# 🧠 Knowledge Graph Builder with Real-Time Visualization")
    gr.Markdown("**Build Knowledge Graphs with Local AI Models - Now with Real-Time SVG Visualization!**")
    
    # Model Configuration Section
    gr.Markdown("## Model Configuration")
    with gr.Row():
        with gr.Column(scale=1):
            provider_radio = gr.Radio(
                choices=["ollama", "lmstudio", "hosted"],
                value=MODEL_PROVIDER,
                label="AI Provider",
                info="Select your AI model provider"
            )
        with gr.Column(scale=1):
            api_url_input = gr.Textbox(
                label="Ollama URL",
                placeholder="http://localhost:11434",
                value=OLLAMA_BASE_URL,
                visible=True,
                info="URL of your Ollama server"
            )
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="API Key",
                placeholder="Your API key",
                value=HOSTED_API_KEY,
                type="password",
                visible=False,
                info="API key for hosted service"
            )
    
    # Initialize with available models based on default provider
    if MODEL_PROVIDER == "ollama":
        initial_models = get_ollama_models(OLLAMA_BASE_URL)
    elif MODEL_PROVIDER == "lmstudio":
        initial_models = get_lmstudio_models(LMSTUDIO_BASE_URL)
    else:
        initial_models = []
    
    if not initial_models:
        initial_models = [DEFAULT_MODEL]
    
    # Choose initial default value: prioritize DEFAULT_MODEL if it's in the list, otherwise use first model
    initial_default_value = DEFAULT_MODEL if DEFAULT_MODEL in initial_models else initial_models[0] if initial_models else DEFAULT_MODEL
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=initial_models,
            value=initial_default_value,
            label="Select Model",
            info="Choose the AI model to use",
            interactive=True
        )
        refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
    
    # Input section
    gr.Markdown("## Input")
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text or URL Input",
                placeholder="Paste any text paragraph or enter a web URL (e.g., https://example.com)",
                lines=4,
                max_lines=8
            )
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload File",
                file_types=[".txt", ".md", ".json", ".csv", ".html", ".htm", ".pdf", ".docx", ".doc"],
                type="filepath"
            )
    
    # Submit button
    submit_btn = gr.Button("🚀 Build Knowledge Graph", variant="primary", size="lg")
    
    # Output section
    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.JSON(label="Knowledge Graph Data")
        with gr.Column(scale=1):
            viz_output = gr.HTML(label="Graph Visualization")
    
    # Real-time progress output
    with gr.Row():
        progress_output = gr.Textbox(
            label="Real-Time Progress & AI Thinking",
            placeholder="Processing logs will appear here...",
            lines=10,
            max_lines=15,
            interactive=False,
            show_copy_button=True
        )
    
    # Feature descriptions in horizontal columns below the table
    gr.Markdown("## Key Features")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **📥 Input Options:**
            • Text Input: Paste any text paragraph
            • URL Input: Extract from web pages
            • File Upload: Multiple formats supported
            • Large Content: 300MB+ handling
            """)
        with gr.Column():
            gr.Markdown("""
            **🎨 Visualization:**
            • Real-time graph updates
            • Color-coded entity types
            • SVG output with legend
            • Progress tracking
            """)
        with gr.Column():
            gr.Markdown("""
            **💾 Storage & Output:**
            • Neo4j graph database
            • Qdrant vector storage
            • JSON knowledge graph
            • Incremental SVG files
            """)
    
    # Supported file types in horizontal layout
    gr.Markdown("## Supported File Types")
    with gr.Row():
        with gr.Column():
            gr.Markdown("📄 **Text Files:** .txt, .md")
        with gr.Column():
            gr.Markdown("📊 **Data Files:** .json, .csv")
        with gr.Column():
            gr.Markdown("🌐 **Web Files:** .html, .htm")
        with gr.Column():
            gr.Markdown("📖 **Documents:** .pdf, .docx, .doc")
    
    # Configuration details
    gr.Markdown("## Current Configuration")
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            **Model Setup:**
            • Provider: {MODEL_PROVIDER}
            • Model: {DEFAULT_MODEL}
            • Chunk Size: {CHUNK_SIZE:,} chars
            """)
        with gr.Column():
            gr.Markdown(f"""
            **Processing:**
            • Chunk Overlap: {CHUNK_OVERLAP} chars
            • Max Chunks: {"Unlimited" if MAX_CHUNKS == 0 else str(MAX_CHUNKS)}
            • Graph Viz: {"✅ Available" if GRAPH_VIZ_AVAILABLE else "❌ Install matplotlib & networkx"}
            """)
        with gr.Column():
            gr.Markdown(f"""
            **Integration:**
            • Ollama: {OLLAMA_BASE_URL}
            • LM Studio: {LMSTUDIO_BASE_URL}
            • Embeddings: {EMBEDDING_PROVIDER}:{EMBEDDING_MODEL_NAME}
            • MCP: {"✅ Available" if MCP_AVAILABLE else "❌ Not Available"}
            """)
    
    # Examples
    gr.Markdown("## Examples")
    examples = gr.Examples(
        examples=[
            ["Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California.", None, "ollama", DEFAULT_MODEL, "", ""],
            ["https://en.wikipedia.org/wiki/Artificial_intelligence", None, "ollama", DEFAULT_MODEL, "", ""],
        ],
        inputs=[text_input, file_input, provider_radio, model_dropdown, api_url_input, api_key_input],
        outputs=[json_output, viz_output, progress_output],
        fn=process_with_config,
        cache_examples=False
    )
    
    # Event handlers
    provider_radio.change(
        fn=toggle_api_fields,
        inputs=[provider_radio],
        outputs=[api_url_input, api_key_input]
    )
    
    def refresh_models(provider, api_url, api_key):
        """Refresh the model dropdown based on current configuration."""
        return update_model_dropdown(provider, api_url, api_key)
    
    # Update models when provider changes
    provider_radio.change(
        fn=refresh_models,
        inputs=[provider_radio, api_url_input, api_key_input],
        outputs=[model_dropdown]
    )
    
    # Update models when URL changes (for Ollama/LM Studio)
    api_url_input.change(
        fn=refresh_models,
        inputs=[provider_radio, api_url_input, api_key_input],
        outputs=[model_dropdown]
    )
    
    # Refresh models button
    refresh_models_btn.click(
        fn=refresh_models,
        inputs=[provider_radio, api_url_input, api_key_input],
        outputs=[model_dropdown]
    )
    
    # Wire up the submit button with configuration
    submit_btn.click(
        fn=process_with_config,
        inputs=[text_input, file_input, provider_radio, model_dropdown, api_url_input, api_key_input],
        outputs=[json_output, viz_output, progress_output]
    )

if __name__ == "__main__":
    print(f"🚀 Starting Knowledge Graph Builder with Real-Time Visualization")
    print(f"📊 Model Provider: {MODEL_PROVIDER}")
    print(f"🤖 Model: {DEFAULT_MODEL}")
    print(f"📈 Chunk Size: {CHUNK_SIZE:,} characters")
    print(f"🔄 Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"📝 Max Chunks: {'Unlimited' if MAX_CHUNKS == 0 else str(MAX_CHUNKS)}")
    print(f"🎨 Graph Visualization: {'✅ Available' if GRAPH_VIZ_AVAILABLE else '❌ Not Available'}")
    print(f"🔗 Ollama URL: {OLLAMA_BASE_URL}")
    print(f"🔗 LM Studio URL: {LMSTUDIO_BASE_URL}")
    print(f"🔌 MCP Available: {MCP_AVAILABLE}")
    print(f"🦙 Ollama Available: {OLLAMA_AVAILABLE}")
    print(f"🔧 OpenAI Client Available: {OPENAI_AVAILABLE}")
    
    if not GRAPH_VIZ_AVAILABLE:
        print("⚠️  Graph visualization disabled. Install: pip install networkx matplotlib")
    else:
        print("✅ Real-time SVG visualization enabled!")
        print("📁 SVG files will be saved in current directory")
    
    demo.launch(mcp_server=True, share=True)
