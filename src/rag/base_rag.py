import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from src.indexing.document_processor import DocumentProcessor


@dataclass
class RAGDocument:
    """Represents a document in the RAG system."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    score: Optional[float] = None


@dataclass
class RAGQuery:
    """Represents a query in the RAG system."""
    query_text: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5


@dataclass
class RAGResult:
    """Represents a search result from the RAG system."""
    documents: List[RAGDocument]
    query: RAGQuery
    metadata: Optional[Dict[str, Any]] = None


class BaseRAG(ABC):
    """Abstract base class for RAG (Retrieval Augmented Generation) implementations."""

    async def add_file(self, file_path: Union[str, Path]) -> str:

        """
        Add a file to the RAG system.

        Args:
            file_path: Path to the file to add

        Returns:
            str: File ID of the added file

        Raises:
            ValueError: If file processing fails
            RuntimeError: If addition fails
        """
        try:
            # Process file into RAG documents
            rag_documents = await self.process_file(file_path)

            # Add documents to system
            doc_ids = await self.add_documents(rag_documents)

            # Return the file ID (first chunk's doc_id as the file identifier)
            return doc_ids[0] if doc_ids else ""

        except Exception as e:
            logging.getLogger(__name__).error(f"Error adding file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to add file: {str(e)}")

    async def process_file(self, file_path: Union[str, Path], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[RAGDocument]:
        """
        Process a file into RAG documents.

        Args:
            file_path: Path to the file to process

        Returns:
            List[RAGDocument]: Processed documents

        Raises:
            ValueError: If file processing fails
            :param file_path:
            :param chunk_overlap:
            :param chunk_size:
        """
        try:
            document_processor = DocumentProcessor()
            # Process file into chunks using existing DocumentProcessor
            processed_chunks = await document_processor.process_file(file_path)

            # Convert ProcessedChunks to RAGDocuments
            rag_documents = []
            for chunk in processed_chunks:
                rag_doc = RAGDocument(
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "processed_at": datetime.now().isoformat()
                    }
                )
                rag_documents.append(rag_doc)

            return rag_documents

        except Exception as e:
            logging.getLogger(__name__).error(f"Error processing file {file_path}: {str(e)}")
            raise ValueError(f"Failed to process file: {str(e)}")

    @abstractmethod
    async def add_documents(self, documents: List[RAGDocument]) -> List[str]:
        """
        Add documents to the RAG system.

        Args:
            documents: List of documents to add

        Returns:
            List[str]: List of document IDs for the added documents

        Raises:
            ValueError: If documents are invalid
            RuntimeError: If addition fails
        """
        pass

    @abstractmethod
    async def search(self, query: RAGQuery) -> RAGResult:
        """
        Search for relevant documents based on a query.

        Args:
            query: Search query parameters

        Returns:
            RAGResult: Search results including relevant documents

        Raises:
            ValueError: If query is invalid
            RuntimeError: If search fails
        """
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the RAG system.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            ValueError: If document IDs are invalid
            RuntimeError: If deletion fails
        """
        pass

    @abstractmethod
    async def update_document(self, doc_id: str, document: RAGDocument) -> bool:
        """
        Update an existing document.

        Args:
            doc_id: ID of document to update
            document: New document content

        Returns:
            bool: True if update was successful

        Raises:
            ValueError: If document ID or content is invalid
            RuntimeError: If update fails
        """
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[RAGDocument]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Optional[RAGDocument]: The document if found, None otherwise

        Raises:
            RuntimeError: If retrieval fails
        """
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all documents from the RAG system.

        Returns:
            bool: True if clearing was successful

        Raises:
            RuntimeError: If clearing fails
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.

        Returns:
            Dict[str, Any]: Statistics about the system

        Raises:
            RuntimeError: If stats collection fails
        """
        pass

    @abstractmethod
    def supports_metadata_search(self) -> bool:
        """
        Whether the RAG implementation supports searching by metadata.

        Returns:
            bool: True if metadata search is supported
        """
        pass

    @abstractmethod
    def get_backend_type(self) -> str:
        """
        Get the type of RAG implementation.

        Returns:
            str: Type identifier for the RAG implementation
        """
        pass

    @abstractmethod
    async def save_state(self, path: Path) -> bool:
        """
        Save the current state of the RAG system.

        Args:
            path: Path to save the state to

        Returns:
            bool: True if save was successful

        Raises:
            RuntimeError: If save fails
        """
        pass

    @abstractmethod
    async def load_state(self, path: Path) -> bool:
        """
        Load a previously saved state.

        Args:
            path: Path to load the state from

        Returns:
            bool: True if load was successful

        Raises:
            RuntimeError: If load fails
            FileNotFoundError: If state file not found
        """
        pass