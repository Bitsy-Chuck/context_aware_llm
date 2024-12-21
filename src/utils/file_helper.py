from typing import List, Dict, Optional, Union
import os
from pathlib import Path
import mimetypes
import magic
import hashlib
from datetime import datetime
import logging

class FileHelper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_mime_types()

    def _initialize_mime_types(self):
        """Initialize additional MIME type mappings."""
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/markdown', '.markdown')
        mimetypes.add_type('application/x-jupyter', '.ipynb')

    def get_file_info(self, file_path: Union[str, Path]) -> Dict:
        """Get detailed information about a file."""
        file_path = Path(file_path)

        try:
            stat = file_path.stat()
            mime_type = magic.from_file(str(file_path), mime=True)

            return {
                "name": file_path.name,
                "extension": file_path.suffix,
                "size": stat.st_size,
                "mime_type": mime_type,
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "hash": self.calculate_file_hash(file_path)
            }
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
            raise

    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_supported_file_type(self, file_path: Union[str, Path]) -> bool:
        """Check if file type is supported."""
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            return mime_type.startswith(('text/', 'application/pdf', 'application/json'))
        except Exception:
            return False

    def ensure_unique_path(self, file_path: Union[str, Path]) -> Path:
        """Ensure file path is unique by adding number suffix if needed."""
        file_path = Path(file_path)
        if not file_path.exists():
            return file_path

        counter = 1
        while True:
            new_path = file_path.parent / f"{file_path.stem}_{counter}{file_path.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1