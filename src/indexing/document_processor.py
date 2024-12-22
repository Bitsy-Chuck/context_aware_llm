from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import mimetypes
import logging
from dataclasses import dataclass
import hashlib
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

from chonkie import TokenChunker, SentenceChunker, SemanticChunker


class ProcessingStrategy(Enum):
    """Enum for different document processing strategies."""
    TOKEN = "token"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: ProcessingStrategy = ProcessingStrategy.HYBRID
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    semantic_similarity_threshold: float = 0.7
    extract_code_elements: bool = True
    extract_metadata: bool = True
    include_document_stats: bool = True


@dataclass
class ProcessedChunk:
    """Represents a processed chunk of text."""
    content: str
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int
    chunk_type: str = "text"  # Can be "text", "code", "mixed"
    code_elements: Optional[Dict[str, List[str]]] = None
    semantic_context: Optional[Dict[str, Any]] = None


class BaseProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    async def process_chunk(self, text: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """Process a chunk of text into smaller chunks."""
        pass


class TokenProcessor(BaseProcessor):
    """Process text using token-based chunking."""

    def __init__(self, config: ProcessingConfig):
        self.chunker = TokenChunker(
            tokenizer="gpt2",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    async def process_chunk(self, text: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        chunks = self.chunker(text)
        return [
            ProcessedChunk(
                content=chunk.text,
                metadata=metadata,
                source_file=metadata["source_file"],
                chunk_index=idx,
                chunk_type="text"
            )
            for idx, chunk in enumerate(chunks)
        ]


class SentenceProcessor(BaseProcessor):
    """Process text using sentence-based chunking."""

    def __init__(self, config: ProcessingConfig):
        self.chunker = SentenceChunker(
            tokenizer="gpt2",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_sentences_per_chunk=1
        )

    async def process_chunk(self, text: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        chunks = self.chunker(text)
        return [
            ProcessedChunk(
                content=chunk.text,
                metadata=metadata,
                source_file=metadata["source_file"],
                chunk_index=idx,
                chunk_type="text"
            )
            for idx, chunk in enumerate(chunks)
        ]


class SemanticProcessor(BaseProcessor):
    """Process text using semantic chunking."""

    def __init__(self, config: ProcessingConfig):
        self.chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            similarity_window=1,
            chunk_size=config.chunk_size
        )

    async def process_chunk(self, text: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        chunks = self.chunker(text)
        return [
            ProcessedChunk(
                content=chunk.text,
                metadata=metadata,
                source_file=metadata["source_file"],
                chunk_index=idx,
                chunk_type="text",
                # semantic_context={"similarity_score": chunk.similarity_score}
            )
            for idx, chunk in enumerate(chunks)
        ]


class HybridProcessor(BaseProcessor):
    """Process text using a combination of strategies."""

    def __init__(self, config: ProcessingConfig):
        self.sentence_processor = SentenceProcessor(config)
        self.semantic_processor = SemanticProcessor(config)
        self.config = config

    async def process_chunk(self, text: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        # First, split into sentence chunks
        initial_chunks = await self.sentence_processor.process_chunk(text, metadata)

        # Then apply semantic processing to each chunk
        final_chunks = []
        for chunk in initial_chunks:
            semantic_chunks = await self.semantic_processor.process_chunk(chunk.content, chunk.metadata)
            final_chunks.extend(semantic_chunks)

        return final_chunks


class DocumentProcessor:
    """Main document processor class."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the document processor."""
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)

        # Register additional MIME types
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/plain', '.txt')

        # Initialize processors
        self.processors = {
            ProcessingStrategy.TOKEN: TokenProcessor(self.config),
            ProcessingStrategy.SENTENCE: SentenceProcessor(self.config),
            ProcessingStrategy.SEMANTIC: SemanticProcessor(self.config),
            ProcessingStrategy.HYBRID: HybridProcessor(self.config)
        }

    def _extract_code_elements(self, content: str) -> Dict[str, List[str]]:
        """Extract code-related elements from content."""
        import re

        code_elements = {
            "functions": [],
            "classes": [],
            "imports": [],
            "code_blocks": []
        }

        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        code_elements["code_blocks"] = [block.strip('`') for block in code_blocks]

        # Extract function definitions
        functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
        code_elements["functions"] = functions

        # Extract class definitions
        classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]', content)
        code_elements["classes"] = classes

        # Extract imports
        imports = re.findall(r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content)
        code_elements["imports"] = imports

        return code_elements

    def _get_document_stats(self, content: str) -> Dict[str, Any]:
        """Generate statistics about the document."""
        return {
            "total_length": len(content),
            "line_count": len(content.splitlines()),
            "word_count": len(content.split()),
            "has_code": bool(self._extract_code_elements(content)["code_blocks"])
        }

    def _generate_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Generate metadata for a document."""
        metadata = {
            "source_file": str(file_path),
            "file_type": mimetypes.guess_type(str(file_path))[0],
            "processed_at": datetime.now().isoformat(),
            "file_size": file_path.stat().st_size,
            "file_name": file_path.name
        }

        if self.config.include_document_stats:
            metadata["document_stats"] = self._get_document_stats(content)

        if self.config.extract_metadata:
            metadata.update({
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "processing_strategy": self.config.strategy.value
            })

        return metadata

    async def process_file(self, file_path: Union[str, Path]) -> List[ProcessedChunk]:
        """Process a single file into chunks."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Generate metadata
            metadata = self._generate_metadata(file_path, content)

            # Process content using selected strategy
            processor = self.processors[self.config.strategy]
            chunks = await processor.process_chunk(content, metadata)

            # Extract code elements if configured
            if self.config.extract_code_elements:
                for chunk in chunks:
                    chunk.code_elements = self._extract_code_elements(chunk.content)

                    # Determine chunk type based on content
                    if chunk.code_elements["code_blocks"]:
                        chunk.chunk_type = "code" if len(chunk.code_elements["code_blocks"]) == 1 else "mixed"

            self.logger.info(f"Processed file {file_path} into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise ValueError(f"Failed to process file: {str(e)}")