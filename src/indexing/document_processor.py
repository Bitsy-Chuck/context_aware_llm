from typing import List, Dict, Optional, Union
import os
from pathlib import Path
import mimetypes
import logging
from dataclasses import dataclass
import hashlib
from datetime import datetime


@dataclass
class ProcessedChunk:
    content: str
    metadata: Dict
    source_file: str
    chunk_index: int


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        # Register additional MIME types
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/plain', '.txt')

    async def process_file(
            self,
            file_path: Union[str, Path],
            file_id: Optional[str] = None
    ) -> List[ProcessedChunk]:
        """Process a single file into chunks."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate file ID if not provided
        if file_id is None:
            file_id = self._generate_file_id(file_path)

        # Determine file type and processing method
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type is None:
            raise ValueError(f"Unsupported file type: {file_path}")

        try:
            if mime_type.startswith('text/'):
                chunks = await self._process_text_file(file_path)
            else:
                raise ValueError(f"Unsupported MIME type: {mime_type}")

            # Add metadata to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append(ProcessedChunk(
                    content=chunk,
                    metadata={
                        "file_id": file_id,
                        "file_name": file_path.name,
                        "mime_type": mime_type,
                        "processed_at": datetime.now().isoformat(),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap
                    },
                    source_file=str(file_path),
                    chunk_index=i
                ))

            return processed_chunks

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def _process_text_file(self, file_path: Path) -> List[str]:
        """Process a text file into chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Simple chunking strategy
        chunks = []
        start = 0

        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at a sentence or paragraph
            for break_char in ['\n\n', '\n', '. ', ' ']:
                last_break = text[start:end].rfind(break_char)
                if last_break != -1:
                    end = start + last_break + len(break_char)
                    break

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks

    def _generate_file_id(self, file_path: Path) -> str:
        """Generate a unique file ID based on file path and contents."""
        hasher = hashlib.sha256()
        hasher.update(str(file_path).encode())

        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()[:16]