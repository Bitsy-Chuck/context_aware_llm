import uuid
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import networkx as nx
import json
import pickle
import asyncio
import hashlib
from dataclasses import dataclass, field
from asyncio import Lock
import re
from collections import defaultdict

from .base_rag import BaseRAG, RAGDocument, RAGQuery, RAGResult
from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..indexing.document_processor import DocumentProcessor, ProcessedChunk


@dataclass(frozen=True)
class GraphNode:
    """Represents a node in the knowledge graph."""
    doc_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    code_snippets: List[str] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    function_signatures: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    weight: float
    relationship_type: str
    metadata: Dict[str, Any]


class GraphRAG(BaseRAG):
    """Graph-based implementation of RAG optimized for technical documentation."""

    def __init__(
            self,
            embedding_model: BaseEmbeddings,
            db_manager: DatabaseManager,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            similarity_threshold: float = 0.7,
            max_connections: int = 5,
            max_cached_embeddings: int = 10000
    ):
        """Initialize GraphRAG with configuration parameters."""
        super().__init__()
        self.embedding_model = embedding_model
        self.db_manager = db_manager
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_connections = max_connections
        self.max_cached_embeddings = max_cached_embeddings

        # Initialize graph and caches
        self.graph = nx.DiGraph()
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.content_hash_map: Dict[str, str] = {}

        # Technical documentation specific indices
        self.function_index: Dict[str, Set[str]] = defaultdict(set)
        self.api_endpoint_index: Dict[str, Set[str]] = defaultdict(set)

        # Concurrency controls
        self._graph_lock = Lock()
        self._embedding_lock = Lock()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_backup_path(self) -> Path:
        """Get path for state backups."""
        return Path("backups") / f"graph_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings with safety checks."""
        if np.all(embedding1 == 0) or np.all(embedding2 == 0):
            return 0.0
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def _extract_technical_info(self, content: str) -> Dict[str, List[str]]:
        """Extract technical information from content."""
        technical_info = {
            "code_snippets": [],
            "api_endpoints": [],
            "function_signatures": [],
            "dependencies": []
        }

        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        technical_info["code_snippets"].extend([block.strip('`') for block in code_blocks])

        # Extract inline code
        inline_code = re.findall(r'`([^`]+)`', content)
        technical_info["code_snippets"].extend(inline_code)

        # Extract function signatures
        function_patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',  # Python
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',  # JavaScript
            r'public\s+\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',  # Java/C#
            r'async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',  # Async Python
            r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'  # Arrow functions
        ]

        for pattern in function_patterns:
            functions = re.findall(pattern, content)
            technical_info["function_signatures"].extend(functions)

        # Extract API endpoints
        api_patterns = [
            r'@\w+\.route\([\'"]([^\'"]+)[\'"]\)',  # Flask/FastAPI
            r'@RequestMapping\([\'"]([^\'"]+)[\'"]\)',  # Spring
            r'app\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]\)',  # Express/Koa
            r'router\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]\)',  # Express Router
            r'@(Get|Post|Put|Delete|Patch)\([\'"]([^\'"]+)[\'"]\)'  # NestJS/TypeScript decorators
        ]

        for pattern in api_patterns:
            endpoints = re.findall(pattern, content)
            processed_endpoints = []
            for endpoint in endpoints:
                if isinstance(endpoint, tuple):
                    # Handle method + path tuples
                    if len(endpoint) == 2:
                        processed_endpoints.append(f"{endpoint[0].upper()}:{endpoint[1]}")
                else:
                    processed_endpoints.append(endpoint)
            technical_info["api_endpoints"].extend(processed_endpoints)

        # Extract dependencies
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Python imports
            r'from\s+([a-zA-Z_][a-zA-Z0-9_.]+)\s+import',  # Python from imports
            r'require\([\'"]([^\'"]+)[\'"]\)',  # Node.js requires
            r'import.*?[\'"]([^\'"]+)[\'"]',  # ES6 imports
            r'using\s+([a-zA-Z_][a-zA-Z0-9_.]+);',  # C# using statements
            r'import\s+([a-zA-Z_][a-zA-Z0-9_.]+);'  # Java imports
        ]

        for pattern in import_patterns:
            imports = re.findall(pattern, content)
            # Clean and normalize imports
            cleaned_imports = [
                imp.split('.')[0] if '.' in imp and not imp.startswith('.')
                else imp for imp in imports
            ]
            technical_info["dependencies"].extend(cleaned_imports)

        # Remove duplicates while preserving order
        for key in technical_info:
            technical_info[key] = list(dict.fromkeys(technical_info[key]))

        return technical_info

    async def _update_graph_connections(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Update graph connections for a new or updated node."""
        async with self._graph_lock:
            # Find related nodes based on embedding similarity
            related_nodes = []
            for node_id, node_embedding in self.node_embeddings.items():
                if node_id != doc_id:
                    similarity = self._calculate_similarity(embedding, node_embedding)
                    if similarity >= self.similarity_threshold:
                        related_nodes.append((node_id, similarity))

            # Sort by similarity and take top K
            related_nodes.sort(key=lambda x: x[1], reverse=True)
            related_nodes = related_nodes[:self.max_connections]

            # Create edges with metadata
            for related_id, similarity in related_nodes:
                related_metadata = self.graph.nodes[related_id]["metadata"]

                # Calculate technical relationships
                shared_functions = set(metadata.get("function_signatures", [])) & \
                                   set(related_metadata.get("function_signatures", []))
                shared_dependencies = set(metadata.get("dependencies", [])) & \
                                      set(related_metadata.get("dependencies", []))
                shared_endpoints = set(metadata.get("api_endpoints", [])) & \
                                   set(related_metadata.get("api_endpoints", []))

                edge_data = GraphEdge(
                    source_id=doc_id,
                    target_id=related_id,
                    weight=similarity,
                    relationship_type="semantic_similarity",
                    metadata={
                        "similarity_score": similarity,
                        "shared_functions": list(shared_functions),
                        "shared_dependencies": list(shared_dependencies),
                        "shared_endpoints": list(shared_endpoints),
                        "created_at": datetime.now().isoformat()
                    }
                )

                # Add bidirectional edges
                self.graph.add_edge(
                    doc_id,
                    related_id,
                    weight=edge_data.weight,
                    relationship_type=edge_data.relationship_type,
                    metadata=edge_data.metadata
                )
                self.graph.add_edge(
                    related_id,
                    doc_id,
                    weight=edge_data.weight,
                    relationship_type=edge_data.relationship_type,
                    metadata=edge_data.metadata
                )

    def _extract_code_context(self, content: str) -> Dict[str, Any]:
        """Extract code-specific context from content."""
        context = {
            "code_blocks": [],
            "related_functions": [],
            "implementation_details": [],
            "usage_examples": [],
            "dependencies": []
        }

        # Extract complete code blocks with language
        code_block_pattern = r'```(\w*)\n([\s\S]*?)```'
        code_blocks = re.findall(code_block_pattern, content)
        for lang, code in code_blocks:
            context["code_blocks"].append({
                "language": lang.lower() if lang else "unknown",
                "code": code.strip()
            })

        # Extract function definitions with docstrings
        python_func_pattern = r'(def\s+\w+\s*\([^)]*\):(?:\s*"""[\s\S]*?""")?\s*(?:[^\n]+\n?)*)'
        js_func_pattern = r'((?:async\s+)?function\s+\w+\s*\([^)]*\)\s*{(?:[^{}]*|{[^{}]*})*})'

        for pattern in [python_func_pattern, js_func_pattern]:
            functions = re.findall(pattern, content)
            context["related_functions"].extend(functions)

        # Extract implementation notes and comments
        implementation_patterns = [
            r'#\s*(?:TODO|NOTE|FIXME|HACK|XXX):.*',  # Python comments
            r'//\s*(?:TODO|NOTE|FIXME|HACK|XXX):.*',  # JS comments
            r'/\*\s*(?:TODO|NOTE|FIXME|HACK|XXX):[\s\S]*?\*/'  # Multi-line comments
        ]

        for pattern in implementation_patterns:
            notes = re.findall(pattern, content)
            context["implementation_details"].extend(notes)

        # Extract usage examples
        example_patterns = [
            r'(?:Example|Usage):\s*```[\s\S]*?```',
            r'(?:Example|Usage):\s*`[^`]+`'
        ]

        for pattern in example_patterns:
            examples = re.findall(pattern, content)
            context["usage_examples"].extend(examples)

        return context

    async def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the current state of the graph and indices."""
        try:
            if filepath is None:
                filepath = self._get_backup_path()

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            async with self._graph_lock:
                state = {
                    "graph": nx.node_link_data(self.graph),
                    "node_embeddings": {k: v.tolist() for k, v in self.node_embeddings.items()},
                    "content_hash_map": self.content_hash_map,
                    "function_index": {k: list(v) for k, v in self.function_index.items()},
                    "api_endpoint_index": {k: list(v) for k, v in self.api_endpoint_index.items()},
                    "config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "similarity_threshold": self.similarity_threshold,
                        "max_connections": self.max_connections,
                        "max_cached_embeddings": self.max_cached_embeddings
                    }
                }

                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)

                self.logger.info(f"State saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise RuntimeError(f"Failed to save state: {str(e)}")

    def _prune_embedding_cache(self) -> None:
        """
        Prunes the embedding cache to stay within memory limits.
        Removes least recently used embeddings when cache exceeds max_cached_embeddings.
        """
        try:
            if len(self.node_embeddings) > self.max_cached_embeddings:
                # Calculate how many embeddings to remove
                num_to_remove = len(self.node_embeddings) - self.max_cached_embeddings

                # Get node access timestamps from graph metadata
                node_timestamps = []
                for node_id in self.node_embeddings:
                    timestamp = self.graph.nodes[node_id]["metadata"].get("last_accessed",
                                                                          self.graph.nodes[node_id]["metadata"].get(
                                                                              "added_at"))
                    node_timestamps.append((node_id, timestamp))

                # Sort by timestamp (oldest first)
                node_timestamps.sort(key=lambda x: x[1])

                # Remove oldest embeddings
                for node_id, _ in node_timestamps[:num_to_remove]:
                    if node_id in self.node_embeddings:
                        del self.node_embeddings[node_id]

                self.logger.info(f"Pruned {num_to_remove} embeddings from cache")

        except Exception as e:
            self.logger.error(f"Error pruning embedding cache: {str(e)}")
            # Don't raise error since this is a maintenance operation

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the graph."""
        try:
            async with self._graph_lock:
                if doc_id not in self.graph:
                    raise ValueError(f"Document {doc_id} not found in graph")

                # Remove from technical indices
                metadata = self.graph.nodes[doc_id]["metadata"]
                for func in metadata.get("function_signatures", []):
                    self.function_index[func].discard(doc_id)
                for endpoint in metadata.get("api_endpoints", []):
                    self.api_endpoint_index[endpoint].discard(doc_id)

                # Remove from graph and embeddings
                self.graph.remove_node(doc_id)
                if doc_id in self.node_embeddings:
                    del self.node_embeddings[doc_id]

                # Remove from database
                await self.db_manager.delete_indexed_file(doc_id)

                return True

        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            raise RuntimeError(f"Failed to delete document: {str(e)}")

    async def get_technical_summary(self, doc_id: str) -> Dict[str, Any]:
        """Get a technical summary of a document."""
        try:
            if doc_id not in self.graph:
                raise ValueError(f"Document {doc_id} not found in graph")

            node_data = self.graph.nodes[doc_id]
            content = node_data["content"]
            metadata = node_data["metadata"]

            # Extract code context
            code_context = self._extract_code_context(content)

            # Get connected documents
            connected_docs = []
            for neighbor in self.graph.neighbors(doc_id):
                edge_data = self.graph.edges[doc_id, neighbor]
                connected_docs.append({
                    "doc_id": neighbor,
                    "similarity": edge_data["weight"],
                    "shared_functions": edge_data["metadata"].get("shared_functions", []),
                    "shared_dependencies": edge_data["metadata"].get("shared_dependencies", []),
                    "shared_endpoints": edge_data["metadata"].get("shared_endpoints", [])
                })

            return {
                "doc_id": doc_id,
                "code_snippets": metadata.get("code_snippets", []),
                "function_signatures": metadata.get("function_signatures", []),
                "api_endpoints": metadata.get("api_endpoints", []),
                "dependencies": metadata.get("dependencies", []),
                "code_context": code_context,
                "connected_documents": connected_docs,
                "source_file": metadata.get("source_file", ""),
                "last_updated": metadata.get("updated_at", metadata.get("added_at", "")),
                "implementation_details": code_context.get("implementation_details", []),
                "usage_examples": code_context.get("usage_examples", [])
            }

        except Exception as e:
            self.logger.error(f"Error getting technical summary: {str(e)}")
            raise RuntimeError(f"Failed to get technical summary: {str(e)}")

    async def get_implementation_examples(self, function_name: str) -> List[Dict[str, Any]]:
        """Get implementation examples for a specific function."""
        try:
            examples = []
            if function_name in self.function_index:
                for doc_id in self.function_index[function_name]:
                    node_data = self.graph.nodes[doc_id]
                    code_context = self._extract_code_context(node_data["content"])

                    # Find the specific function implementation
                    relevant_functions = [
                        f for f in code_context["related_functions"]
                        if function_name in f
                    ]

                    if relevant_functions:
                        examples.append({
                            "doc_id": doc_id,
                            "implementations": relevant_functions,
                            "usage_examples": [
                                ex for ex in code_context["usage_examples"]
                                if function_name in ex
                            ],
                            "source_file": node_data["metadata"].get("source_file", ""),
                            "dependencies": node_data["metadata"].get("dependencies", [])
                        })

            return examples

        except Exception as e:
            self.logger.error(f"Error getting implementation examples: {str(e)}")
            raise RuntimeError(f"Failed to get implementation examples: {str(e)}")

    async def get_related_endpoints(self, function_name: str) -> List[Dict[str, Any]]:
        """Get related API endpoints for a specific function."""
        try:
            related_endpoints = []
            if function_name in self.function_index:
                function_nodes = self.function_index[function_name]

                # Get all nodes connected to function nodes
                connected_nodes = set()
                for node_id in function_nodes:
                    connected_nodes.update(self.graph.neighbors(node_id))

                # Find endpoints in connected nodes
                for node_id in connected_nodes:
                    node_data = self.graph.nodes[node_id]
                    endpoints = node_data["metadata"].get("api_endpoints", [])
                    if endpoints:
                        related_endpoints.append({
                            "doc_id": node_id,
                            "endpoints": endpoints,
                            "relationship": self.graph.edges[list(function_nodes)[0], node_id]["relationship_type"],
                            "similarity": self.graph.edges[list(function_nodes)[0], node_id]["weight"]
                        })

            return sorted(related_endpoints, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error getting related endpoints: {str(e)}")
            raise RuntimeError(f"Failed to get related endpoints: {str(e)}")

    async def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency relationships across all documents."""
        try:
            dependency_analysis = {
                "dependency_frequency": defaultdict(int),
                "dependency_clusters": defaultdict(list),
                "central_dependencies": [],
                "isolated_dependencies": []
            }

            # Calculate dependency frequencies and relationships
            for node_id in self.graph.nodes():
                node_deps = self.graph.nodes[node_id]["metadata"].get("dependencies", [])
                for dep in node_deps:
                    dependency_analysis["dependency_frequency"][dep] += 1
                    dependency_analysis["dependency_clusters"][dep].append(node_id)

            # Calculate centrality of dependencies
            dep_subgraph = nx.Graph()
            for node_id in self.graph.nodes():
                deps = self.graph.nodes[node_id]["metadata"].get("dependencies", [])
                for dep1 in deps:
                    for dep2 in deps:
                        if dep1 != dep2:
                            if not dep_subgraph.has_edge(dep1, dep2):
                                dep_subgraph.add_edge(dep1, dep2, weight=1)
                            else:
                                dep_subgraph.edges[dep1, dep2]["weight"] += 1

            if dep_subgraph.nodes:
                # Calculate eigenvector centrality
                centrality = nx.eigenvector_centrality_numpy(dep_subgraph, weight="weight")
                central_deps = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]

                dependency_analysis["central_dependencies"] = [
                    {
                        "dependency": dep,
                        "centrality": cent,
                        "frequency": dependency_analysis["dependency_frequency"][dep],
                        "connected_dependencies": list(dep_subgraph.neighbors(dep))
                    }
                    for dep, cent in central_deps
                ]

                # Find isolated dependencies
                dependency_analysis["isolated_dependencies"] = [
                    dep for dep in dependency_analysis["dependency_frequency"]
                    if dep not in dep_subgraph or dep_subgraph.degree(dep) == 0
                ]

            return dict(dependency_analysis)

        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {str(e)}")
            raise RuntimeError(f"Failed to analyze dependencies: {str(e)}")

    async def get_document_hierarchy(self) -> Dict[str, Any]:
        """Generate a hierarchical view of documents based on their relationships."""
        try:
            hierarchy = {
                "root_documents": [],
                "dependency_trees": [],
                "implementation_groups": defaultdict(list),
                "isolated_documents": []
            }

            # Calculate document centrality
            centrality = nx.eigenvector_centrality_numpy(self.graph, weight="weight")

            # Identify root documents (high centrality, many connections)
            root_docs = sorted(
                centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for doc_id, cent in root_docs:
                doc_data = self.graph.nodes[doc_id]
                hierarchy["root_documents"].append({
                    "doc_id": doc_id,
                    "centrality": cent,
                    "title": doc_data["metadata"].get("title", ""),
                    "source_file": doc_data["metadata"].get("source_file", ""),
                    "connected_docs": len(list(self.graph.neighbors(doc_id)))
                })

            # Build dependency trees
            for root_doc_id, _ in root_docs:
                tree = {
                    "root": root_doc_id,
                    "dependencies": [],
                    "implementations": []
                }

                # Get immediate dependencies
                for neighbor in self.graph.neighbors(root_doc_id):
                    edge_data = self.graph.edges[root_doc_id, neighbor]
                    if edge_data["relationship_type"] == "dependency":
                        tree["dependencies"].append({
                            "doc_id": neighbor,
                            "weight": edge_data["weight"],
                            "shared_dependencies": edge_data["metadata"].get("shared_dependencies", [])
                        })

                # Get implementations
                root_funcs = self.graph.nodes[root_doc_id]["metadata"].get("function_signatures", [])
                for func in root_funcs:
                    if func in self.function_index:
                        tree["implementations"].extend(list(self.function_index[func]))

                hierarchy["dependency_trees"].append(tree)

            # Group by implementation patterns
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                funcs = node_data["metadata"].get("function_signatures", [])
                for func in funcs:
                    hierarchy["implementation_groups"][func].append(node_id)

            # Find isolated documents
            hierarchy["isolated_documents"] = [
                node_id for node_id in self.graph.nodes()
                if self.graph.degree(node_id) == 0
            ]

            return dict(hierarchy)

        except Exception as e:
            self.logger.error(f"Error generating document hierarchy: {str(e)}")
            raise RuntimeError(f"Failed to generate document hierarchy: {str(e)}")

    async def optimize_graph(self) -> None:
        """Optimize the graph structure and clean up unnecessary connections."""
        try:
            async with self._graph_lock:
                # Remove weak connections
                edges_to_remove = [
                    (u, v) for u, v, d in self.graph.edges(data=True)
                    if d["weight"] < self.similarity_threshold
                ]
                self.graph.remove_edges_from(edges_to_remove)

                # Remove duplicate edges
                simplified_edges = {}
                for u, v, data in self.graph.edges(data=True):
                    edge_key = tuple(sorted([u, v]))
                    if edge_key not in simplified_edges:
                        simplified_edges[edge_key] = data
                    else:
                        # Keep the stronger connection
                        if data["weight"] > simplified_edges[edge_key]["weight"]:
                            simplified_edges[edge_key] = data

                # Rebuild graph with optimized edges
                new_graph = nx.DiGraph()
                new_graph.add_nodes_from(self.graph.nodes(data=True))
                for (u, v), data in simplified_edges.items():
                    new_graph.add_edge(u, v, **data)
                    new_graph.add_edge(v, u, **data)

                self.graph = new_graph

                # Clean up technical indices
                for index in [self.function_index, self.api_endpoint_index]:
                    keys_to_remove = []
                    for key, doc_ids in index.items():
                        # Remove references to non-existent documents
                        doc_ids = {doc_id for doc_id in doc_ids if doc_id in self.graph}
                        if doc_ids:
                            index[key] = doc_ids
                        else:
                            keys_to_remove.append(key)

                    for key in keys_to_remove:
                        del index[key]

                # Prune embedding cache
                self._prune_embedding_cache()

                self.logger.info("Graph optimization completed successfully")

        except Exception as e:
            self.logger.error(f"Error optimizing graph: {str(e)}")
            raise RuntimeError(f"Failed to optimize graph: {str(e)}")

    async def add_documents(self, documents: List[RAGDocument]) -> List[str]:
        """Add multiple documents to the graph."""
        try:
            doc_ids = []

            for doc in documents:
                # Generate document ID if not provided
                if not doc.doc_id:
                    doc.doc_id = str(uuid.uuid4())

                # Calculate checksum for deduplication
                content_hash = self._calculate_checksum(doc.content)

                # Check for duplicates
                if content_hash in self.content_hash_map:
                    self.logger.warning(f"Duplicate content detected for {doc.doc_id}")
                    doc_ids.append(self.content_hash_map[content_hash])
                    continue

                # Generate embedding if not provided
                embedding = doc.embedding if doc.embedding else \
                    await self.embedding_model.embed_texts([doc.content])[0]

                # Extract technical information
                technical_info = self._extract_technical_info(doc.content)

                # Create node
                node = GraphNode(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    embedding=embedding,
                    metadata={**doc.metadata, **technical_info, "added_at": datetime.now().isoformat()},
                    code_snippets=technical_info["code_snippets"],
                    api_endpoints=technical_info["api_endpoints"],
                    function_signatures=technical_info["function_signatures"],
                    dependencies=technical_info["dependencies"]
                )

                async with self._graph_lock:
                    # Add to graph
                    self.graph.add_node(
                        node.doc_id,
                        content=node.content,
                        metadata=node.metadata
                    )
                    self.node_embeddings[node.doc_id] = node.embedding
                    self.content_hash_map[content_hash] = node.doc_id

                    # Update technical indices
                    for func in node.function_signatures:
                        self.function_index[func].add(node.doc_id)
                    for endpoint in node.api_endpoints:
                        self.api_endpoint_index[endpoint].add(node.doc_id)

                    # Update graph connections
                    await self._update_graph_connections(
                        node.doc_id,
                        node.embedding,
                        node.metadata
                    )

                    doc_ids.append(node.doc_id)

                    # Update database
                    await self.db_manager.save_indexed_file(
                        file_id=node.doc_id,
                        file_path=node.metadata.get("source_file", "unknown"),
                        file_type=node.metadata.get("mime_type", "text/plain"),
                        metadata=node.metadata,
                        embedding_model=self.embedding_model.get_model_name(),
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )

            return doc_ids

        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise RuntimeError(f"Failed to add documents: {str(e)}")

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete multiple documents from the graph."""
        try:
            async with self._graph_lock:
                for doc_id in doc_ids:
                    if doc_id not in self.graph:
                        self.logger.warning(f"Document {doc_id} not found in graph")
                        continue

                    # Remove from technical indices
                    metadata = self.graph.nodes[doc_id]["metadata"]
                    for func in metadata.get("function_signatures", []):
                        self.function_index[func].discard(doc_id)
                    for endpoint in metadata.get("api_endpoints", []):
                        self.api_endpoint_index[endpoint].discard(doc_id)

                    # Remove from graph and embeddings
                    self.graph.remove_node(doc_id)
                    if doc_id in self.node_embeddings:
                        del self.node_embeddings[doc_id]

                    # Remove from database
                    await self.db_manager.delete_indexed_file(doc_id)

                return True

        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}")

    async def get_document(self, doc_id: str) -> Optional[RAGDocument]:
        """Retrieve a specific document by ID."""
        try:
            if doc_id not in self.graph:
                return None

            node_data = self.graph.nodes[doc_id]
            return RAGDocument(
                doc_id=doc_id,
                content=node_data["content"],
                metadata=node_data["metadata"],
                embedding=self.node_embeddings.get(doc_id, None).tolist()
                if doc_id in self.node_embeddings else None
            )

        except Exception as e:
            self.logger.error(f"Error retrieving document: {str(e)}")
            raise RuntimeError(f"Failed to retrieve document: {str(e)}")

    async def clear(self) -> bool:
        pass
        """Clear all documents and reset the graph."""
        try:
            async with self._graph_lock:
                # Clear graph and indices
                self.graph.clear()
                self.node_embeddings.clear()
                self.content_hash_map.clear()
                self.function_index.clear()
                self.api_endpoint_index.clear()

                # Clear database
                await self.db_manager.clear_indexed_files()

                return True

        except Exception as e:
            self.logger.error(f"Error clearing graph: {str(e)}")
            raise RuntimeError(f"Failed to clear graph: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph-based RAG system."""
        try:
            stats = {
                "total_documents": self.graph.number_of_nodes(),
                "total_connections": self.graph.number_of_edges(),
                "average_connections_per_document": (
                    self.graph.number_of_edges() / self.graph.number_of_nodes()
                    if self.graph.number_of_nodes() > 0 else 0
                ),
                "indexed_functions": len(self.function_index),
                "indexed_endpoints": len(self.api_endpoint_index),
                "cached_embeddings": len(self.node_embeddings),
                "graph_density": nx.density(self.graph),
                "strongly_connected_components": nx.number_strongly_connected_components(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
                "technical_indices": {
                    "functions": {
                        "total": sum(len(docs) for docs in self.function_index.values()),
                        "unique": len(self.function_index)
                    },
                    "endpoints": {
                        "total": sum(len(docs) for docs in self.api_endpoint_index.values()),
                        "unique": len(self.api_endpoint_index)
                    }
                }
            }
            return stats

        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            raise RuntimeError(f"Failed to get stats: {str(e)}")

    def supports_metadata_search(self) -> bool:
        """Check if metadata search is supported."""
        return True

    def get_backend_type(self) -> str:
        """Get the RAG implementation type."""
        return "graph_rag"

    async def update_document(self, doc_id: str, document: RAGDocument) -> bool:
        pass
        """Update an existing document with new RAGDocument content."""
        try:
            async with self._graph_lock:
                if doc_id not in self.graph:
                    raise ValueError(f"Document {doc_id} not found in graph")

                # Calculate new embedding if not provided
                new_embedding = document.embedding if document.embedding else \
                    await self.embedding_model.embed_texts([document.content])[0]

                # Extract technical information
                technical_info = self._extract_technical_info(document.content)

                # Update node
                old_metadata = self.graph.nodes[doc_id]["metadata"]
                new_metadata = {
                    **document.metadata,
                    **technical_info,
                    "updated_at": datetime.now().isoformat()
                }

                self.graph.nodes[doc_id]["content"] = document.content
                self.graph.nodes[doc_id]["metadata"] = new_metadata
                self.node_embeddings[doc_id] = new_embedding

                # Update technical indices
                # Remove old entries
                for func in old_metadata.get("function_signatures", []):
                    self.function_index[func].discard(doc_id)
                for endpoint in old_metadata.get("api_endpoints", []):
                    self.api_endpoint_index[endpoint].discard(doc_id)

                # Add new entries
                for func in technical_info["function_signatures"]:
                    self.function_index[func].add(doc_id)
                for endpoint in technical_info["api_endpoints"]:
                    self.api_endpoint_index[endpoint].add(doc_id)

                # Update graph connections
                await self._update_graph_connections(doc_id, new_embedding, new_metadata)

                # Update database
                await self.db_manager.update_indexed_file(
                    file_id=doc_id,
                    metadata=new_metadata
                )

                return True

        except Exception as e:
            self.logger.error(f"Error updating document: {str(e)}")
            raise RuntimeError(f"Failed to update document: {str(e)}")

    async def search(self, query: RAGQuery) -> RAGResult:
        """
        Search for relevant documents using graph traversal.
        Optimized for code generation queries by prioritizing:
        - Code snippets and examples
        - Function implementations
        - API documentation
        - Usage patterns
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.embed_query(query.query_text)

            async with self._graph_lock:
                # Initial similarity-based node ranking
                initial_rankings = []
                for node_id, node_embedding in self.node_embeddings.items():
                    similarity = self._calculate_similarity(query_embedding, node_embedding)
                    if similarity >= self.similarity_threshold:
                        initial_rankings.append((node_id, similarity))

                # Sort by similarity
                initial_rankings.sort(key=lambda x: x[1], reverse=True)

                # Apply metadata filters if provided
                if query.filters:
                    filtered_rankings = [
                        (node_id, score) for node_id, score in initial_rankings
                        if all(
                            self.graph.nodes[node_id]["metadata"].get(k) == v
                            for k, v in query.filters.items()
                        )
                    ]
                    initial_rankings = filtered_rankings

                # Code generation specific scoring boost
                code_keywords = {
                    "code", "function", "api", "endpoint", "implementation",
                    "example", "snippet", "write", "generate", "usage",
                    "documentation", "method", "class", "interface", "chart",
                    "visualization", "graph", "plot", "component"
                }

                query_lower = query.query_text.lower()
                is_code_query = any(keyword in query_lower for keyword in code_keywords)

                if is_code_query:
                    # Extract technical information from query
                    query_tech_info = self._extract_technical_info(query.query_text)

                    # Find nodes with matching technical attributes
                    technical_matches = set()

                    # Match function signatures
                    for func in query_tech_info["function_signatures"]:
                        technical_matches.update(self.function_index[func])

                    # Match API endpoints
                    for endpoint in query_tech_info["api_endpoints"]:
                        technical_matches.update(self.api_endpoint_index[endpoint])

                    # Match dependencies
                    dependency_matches = set()
                    for dep in query_tech_info["dependencies"]:
                        for node_id in self.graph.nodes():
                            if dep in self.graph.nodes[node_id]["metadata"].get("dependencies", []):
                                dependency_matches.add(node_id)

                    technical_matches.update(dependency_matches)

                    # Boost rankings for technical matches
                    if technical_matches:
                        boosted_rankings = []
                        for node_id, score in initial_rankings:
                            boost_multiplier = 1.0
                            node_data = self.graph.nodes[node_id]

                            # Boost code examples and implementations
                            if node_id in technical_matches:
                                boost_multiplier += 0.5

                            # Additional boost for code blocks
                            if len(node_data["metadata"].get("code_snippets", [])) > 0:
                                boost_multiplier += 0.3

                            # Boost for usage examples
                            code_context = self._extract_code_context(node_data["content"])
                            if len(code_context["usage_examples"]) > 0:
                                boost_multiplier += 0.2

                            boosted_rankings.append((node_id, score * boost_multiplier))
                        initial_rankings = sorted(boosted_rankings, key=lambda x: x[1], reverse=True)

                # Get top results and perform graph traversal
                top_k = min(5, len(initial_rankings))
                if top_k == 0:
                    return RAGResult(
                        documents=[],
                        query=query,
                        metadata={"scores": [], "query_type": "code" if is_code_query else "general"}
                    )

                seed_nodes = [node_id for node_id, _ in initial_rankings[:top_k]]

                # Perform personalized PageRank from seed nodes
                personalization = {node: 1.0 if node in seed_nodes else 0.0
                                   for node in self.graph.nodes()}
                pagerank_scores = nx.pagerank(
                    self.graph,
                    personalization=personalization,
                    weight='weight'
                )

                # Combine initial similarity with graph scores
                final_scores = {}
                for node_id in self.graph.nodes():
                    initial_score = next((score for nid, score in initial_rankings
                                          if nid == node_id), 0.0)
                    graph_score = pagerank_scores.get(node_id, 0.0)
                    final_scores[node_id] = 0.7 * initial_score + 0.3 * graph_score

                # Sort by final scores
                ranked_results = sorted(
                    final_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:query.top_k if query.top_k else 10]

                # Prepare result documents
                result_documents = []
                result_scores = []

                for node_id, score in ranked_results:
                    node_data = self.graph.nodes[node_id]

                    # Extract code context for code queries
                    if is_code_query:
                        code_context = self._extract_code_context(node_data["content"])
                        node_data["metadata"]["code_context"] = code_context

                    # Create RAG document with enriched metadata
                    doc = RAGDocument(
                        doc_id=node_id,
                        content=node_data["content"],
                        metadata={
                            **node_data["metadata"],
                            "relevance_score": score,
                            "has_code_examples": bool(code_context["usage_examples"]) if is_code_query else False,
                            "has_implementation": bool(code_context["related_functions"]) if is_code_query else False
                        }
                    )
                    result_documents.append(doc)
                    result_scores.append(score)

                return RAGResult(
                    documents=result_documents,
                    query=query,
                    metadata={
                        "scores": result_scores,
                        "query_type": "code" if is_code_query else "general",
                        "technical_matches": len(technical_matches) if is_code_query else 0,
                        "total_code_examples": sum(1 for doc in result_documents
                                                   if doc.metadata.get("has_code_examples", False)),
                        "total_implementations": sum(1 for doc in result_documents
                                                     if doc.metadata.get("has_implementation", False))
                    }
                )

        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")

    async def save_state(self, filepath: Optional[Path] = None) -> bool:
        """Save the current state of the graph and indices."""
        try:
            if filepath is None:
                filepath = self._get_backup_path()

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            async with self._graph_lock:
                state = {
                    "graph": nx.node_link_data(self.graph),
                    "node_embeddings": {k: v.tolist() for k, v in self.node_embeddings.items()},
                    "content_hash_map": self.content_hash_map,
                    "function_index": {k: list(v) for k, v in self.function_index.items()},
                    "api_endpoint_index": {k: list(v) for k, v in self.api_endpoint_index.items()},
                    "config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "similarity_threshold": self.similarity_threshold,
                        "max_connections": self.max_connections,
                        "max_cached_embeddings": self.max_cached_embeddings
                    }
                }

                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)

                self.logger.info(f"State saved to {filepath}")
                return True

        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            return False

    async def load_state(self, filepath: Path) -> bool:
        """Load a previously saved state."""
        try:
            async with self._graph_lock:
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)

                # Restore graph
                self.graph = nx.node_link_graph(state["graph"])

                # Restore embeddings
                self.node_embeddings = {k: np.array(v) for k, v in state["node_embeddings"].items()}

                # Restore indices
                self.content_hash_map = state["content_hash_map"]
                self.function_index = {k: set(v) for k, v in state["function_index"].items()}
                self.api_endpoint_index = {k: set(v) for k, v in state["api_endpoint_index"].items()}

                # Restore configuration
                config = state["config"]
                self.chunk_size = config["chunk_size"]
                self.chunk_overlap = config["chunk_overlap"]
                self.similarity_threshold = config["similarity_threshold"]
                self.max_connections = config["max_connections"]
                self.max_cached_embeddings = config["max_cached_embeddings"]

                self.logger.info(f"State loaded from {filepath}")
                return True

        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            return False
