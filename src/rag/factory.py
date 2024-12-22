from typing import Dict, Any, Optional, Type
from pathlib import Path
import logging
from enum import Enum

from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from .base_rag import BaseRAG
from .vector_rag import VectorRAG
from .graph_rag import GraphRAG


class RAGType(Enum):
    """Enum for different types of RAG implementations."""
    VECTOR = "vector"
    GRAPH = "graph"
    # Add other RAG types here as they are implemented


class RAGFactory:
    """Factory for creating and configuring RAG implementations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._rag_implementations: Dict[RAGType, Type[BaseRAG]] = {
            RAGType.VECTOR: VectorRAG,
            RAGType.GRAPH: GraphRAG,
            # Add other implementations here
        }

    async def create_rag(
            self,
            rag_type: RAGType,
            embedding_model: BaseEmbeddings,
            db_manager: DatabaseManager,
            vector_store: Optional[VectorStore] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> BaseRAG:
        """
        Create a RAG implementation with the specified configuration.

        Args:
            rag_type: Type of RAG implementation to create
            embedding_model: Model for generating embeddings
            db_manager: Database manager instance
            vector_store: Optional vector store (required for VectorRAG)
            config: Additional configuration parameters

        Returns:
            BaseRAG: Configured RAG implementation

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If RAG creation fails
        """
        try:
            # Get default configuration
            default_config = self._get_default_config(rag_type)

            # Merge with provided config
            final_config = {**default_config, **(config or {})}

            # Validate configuration
            self._validate_config(rag_type, final_config, vector_store)

            # Get RAG implementation class
            if rag_type not in self._rag_implementations:
                raise ValueError(f"Unsupported RAG type: {rag_type}")

            rag_class = self._rag_implementations[rag_type]

            # Initialize RAG implementation with appropriate parameters
            if rag_type == RAGType.VECTOR:
                if not vector_store:
                    raise ValueError("VectorStore is required for VectorRAG")

                rag = rag_class(
                    embedding_model=embedding_model,
                    vector_store=vector_store,
                    db_manager=db_manager,
                    chunk_size=final_config["chunk_size"],
                    chunk_overlap=final_config["chunk_overlap"]
                )

            elif rag_type == RAGType.GRAPH:
                rag = rag_class(
                    embedding_model=embedding_model,
                    db_manager=db_manager,
                    chunk_size=final_config["chunk_size"],
                    chunk_overlap=final_config["chunk_overlap"],
                    similarity_threshold=final_config["similarity_threshold"],
                    max_connections=final_config["max_connections"],
                    max_cached_embeddings=final_config["max_cached_embeddings"]
                )

            else:
                raise ValueError(f"Implementation not found for RAG type: {rag_type}")

            # Initialize state if path provided
            if "state_path" in final_config:
                state_path = Path(final_config["state_path"])
                if state_path.exists():
                    await rag.load_state(state_path)

            self.logger.info(f"Successfully created {rag_type.value} RAG implementation")
            return rag

        except Exception as e:
            self.logger.error(f"Error creating RAG implementation: {str(e)}")
            raise RuntimeError(f"Failed to create RAG implementation: {str(e)}")

    def _get_default_config(self, rag_type: RAGType) -> Dict[str, Any]:
        """Get default configuration for RAG type."""
        base_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }

        if rag_type == RAGType.VECTOR:
            return base_config

        elif rag_type == RAGType.GRAPH:
            return {
                **base_config,
                "similarity_threshold": 0.7,
                "max_connections": 5,
                "max_cached_embeddings": 10000
            }

        return base_config

    def _validate_config(
            self,
            rag_type: RAGType,
            config: Dict[str, Any],
            vector_store: Optional[VectorStore]
    ) -> None:
        """
        Validate configuration for RAG type.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate base configuration
        if config.get("chunk_size", 0) <= 0:
            raise ValueError("chunk_size must be positive")
        if config.get("chunk_overlap", 0) < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if config.get("chunk_overlap", 0) >= config.get("chunk_size", 0):
            raise ValueError("chunk_overlap must be less than chunk_size")

        # Validate type-specific configuration
        if rag_type == RAGType.VECTOR:
            if not vector_store:
                raise ValueError("VectorStore is required for VectorRAG")

        elif rag_type == RAGType.GRAPH:
            if not 0 <= config.get("similarity_threshold", 0) <= 1:
                raise ValueError("similarity_threshold must be between 0 and 1")
            if config.get("max_connections", 0) <= 0:
                raise ValueError("max_connections must be positive")
            if config.get("max_cached_embeddings", 0) <= 0:
                raise ValueError("max_cached_embeddings must be positive")

    def register_rag_implementation(
            self,
            rag_type: RAGType,
            implementation: Type[BaseRAG]
    ) -> None:
        """
        Register a new RAG implementation.

        Args:
            rag_type: Type identifier for the implementation
            implementation: RAG implementation class
        """
        if not issubclass(implementation, BaseRAG):
            raise ValueError("Implementation must inherit from BaseRAG")

        self._rag_implementations[rag_type] = implementation
        self.logger.info(f"Registered new RAG implementation: {rag_type.value}")

    def get_supported_types(self) -> list[RAGType]:
        """Get list of supported RAG types."""
        return list(self._rag_implementations.keys())

    def get_implementation_info(self, rag_type: RAGType) -> Dict[str, Any]:
        """
        Get information about a RAG implementation.

        Args:
            rag_type: Type of RAG implementation

        Returns:
            Dict with implementation details
        """
        if rag_type not in self._rag_implementations:
            raise ValueError(f"Unsupported RAG type: {rag_type}")

        implementation = self._rag_implementations[rag_type]
        default_config = self._get_default_config(rag_type)

        return {
            "name": rag_type.value,
            "class": implementation.__name__,
            "default_config": default_config,
            "supports_metadata_search": True,  # All current implementations support this
            "description": implementation.__doc__ or "No description available"
        }