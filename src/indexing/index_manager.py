import traceback
from typing import List, Dict, Optional, Union
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from .document_processor import DocumentProcessor, ProcessedChunk


@dataclass
class IndexedFile:
    file_id: str
    file_path: str
    file_type: str
    num_chunks: int
    vector_ids: List[int]  # Changed from List[str] to List[int] to match VectorStore
    embedding_model: str
    indexed_at: datetime
    metadata: Dict


class IndexManager:
    def __init__(
            self,
            embedding_model: BaseEmbeddings,
            db_manager: DatabaseManager,
            vector_store: VectorStore,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        self.embedding_model = embedding_model
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.logger = logging.getLogger(__name__)

    async def index_file(self, file_path: Union[str, Path]) -> str:
        """
        Index a single file.

        Args:
            file_path: Path to the file to be indexed

        Returns:
            str: The file_id of the indexed file

        Raises:
            ValueError: If no content could be extracted from the file
            Exception: If any error occurs during indexing
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")

            # Process the file into chunks
            chunks = await self.document_processor.process_file(file_path)
            if not chunks:
                raise ValueError(f"No content extracted from file: {file_path}")

            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_model.embed_texts(texts)

            # Store in vector database
            metadata = [chunk.metadata for chunk in chunks]
            vector_ids = await self.vector_store.add_embeddings(
                embeddings=embeddings,
                texts=texts,
                metadata=metadata
            )

            # Prepare file information
            file_info = IndexedFile(
                file_id=chunks[0].metadata["file_id"],
                file_path=str(file_path),
                file_type=chunks[0].metadata["mime_type"],
                num_chunks=len(chunks),
                vector_ids=vector_ids,  # These are now integers from VectorStore
                embedding_model=self.embedding_model.get_model_name(),
                indexed_at=datetime.now(),
                metadata={
                    "chunk_size": self.document_processor.chunk_size,
                    "chunk_overlap": self.document_processor.chunk_overlap,
                    "file_size": file_path.stat().st_size,
                    "vector_dimension": self.vector_store.dimension
                }
            )

            # Save to database
            await self.db_manager.save_indexed_file(**asdict(file_info))

            self.logger.info(f"Successfully indexed file: {file_path} with ID: {file_info.file_id}")
            return file_info.file_id

        except Exception as e:
            self.logger.error(f"Error indexing file {file_path}: {str(e)}")
            self.logger.error(f"Stacktrace: {traceback.format_exc()}")
            raise

    async def search_similar(
            self,
            query: str,
            k: int = 5,
            filter_criteria: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar content using the query.

        Args:
            query: The search query
            k: Number of results to return
            filter_criteria: Optional filters for the search

        Returns:
            List[Dict]: List of search results with metadata and scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_model.embed_query(query)

            # Search in vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                filter_criteria=filter_criteria
            )

            # Format results to match VectorStore output
            formatted_results = []
            for idx, distance, metadata in results:
                formatted_results.append({
                    "content": metadata["text"],
                    "metadata": {
                        **metadata["metadata"],
                        "similarity_score": 1.0 / (1.0 + distance),  # Convert distance to similarity
                        "vector_id": idx,
                        "distance": distance
                    }
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            self.logger.error(f"Stacktrace: {traceback.format_exc()}")
            raise

    async def get_indexed_files_stats(self) -> List[Dict]:
        """
        Get statistics about indexed files.

        Returns:
            List[Dict]: List of indexed files with their statistics
        """
        try:
            files = await self.db_manager.get_indexed_files()

            enhanced_files = []
            for file in files:
                try:
                    file_path = Path(file["file_path"])
                    stats = {
                        **file,
                        "exists": file_path.exists(),
                        "size": file_path.stat().st_size if file_path.exists() else None,
                        "last_modified": datetime.fromtimestamp(
                            file_path.stat().st_mtime) if file_path.exists() else None,
                        "vector_store_stats": {
                            "total_vectors": len(file["vector_ids"]) if "vector_ids" in file else 0,
                            "dimension": self.vector_store.dimension
                        }
                    }
                    enhanced_files.append(stats)
                except Exception as e:
                    self.logger.warning(f"Error getting stats for file {file['file_id']}: {str(e)}")
                    enhanced_files.append(file)

            return enhanced_files

        except Exception as e:
            self.logger.error(f"Error getting indexed files stats: {str(e)}")
            self.logger.error(f"Stacktrace: {traceback.format_exc()}")
            raise