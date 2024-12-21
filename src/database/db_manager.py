from typing import List, Dict, Optional
import asyncpg
import json
from datetime import datetime
import os
import logging
from pathlib import Path
import asyncio
from urllib.parse import urlparse


class DatabaseManager:
    def __init__(self, db_url: str = None):
        """
        Initialize DatabaseManager with PostgreSQL connection.

        Args:
            db_url: PostgreSQL connection URL in format:
                   postgresql://user:password@host:port/dbname
        """
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("Database URL must be provided either in constructor or DATABASE_URL environment variable")

        self.pool = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize database connection pool and tables."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.db_url)
            await self._init_db()

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()

    async def _init_db(self):
        """Initialize database tables."""
        async with self.pool.acquire() as conn:
            # Create chats table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP WITH TIME ZONE,
                    last_updated TIMESTAMP WITH TIME ZONE
                )
            """)

            # Create messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT REFERENCES chats(chat_id) ON DELETE CASCADE,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    metadata JSONB
                )
            """)

            # Create indexed_files table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    file_type TEXT,
                    indexed_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB,
                    embedding_model TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER
                )
            """)

            # Create indices for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_indexed_files_path ON indexed_files(file_path);
                CREATE INDEX IF NOT EXISTS idx_chats_last_updated ON chats(last_updated);
            """)

    async def create_chat(self, chat_id: str, title: str) -> None:
        """Create a new chat session."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chats (chat_id, title, created_at, last_updated)
                VALUES ($1, $2, NOW(), NOW())
            """, chat_id, title)

    async def save_message(
            self,
            chat_id: str,
            message_id: str,
            role: str,
            content: str,
            metadata: Dict = None
    ) -> None:
        """Save a message to the database."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Save message
                await conn.execute("""
                    INSERT INTO messages (message_id, chat_id, role, content, timestamp, metadata)
                    VALUES ($1, $2, $3, $4, NOW(), $5)
                """, message_id, chat_id, role, content, json.dumps(metadata) if metadata else None)

                # Update chat last_updated
                await conn.execute("""
                    UPDATE chats SET last_updated = NOW()
                    WHERE chat_id = $1
                """, chat_id)

    async def get_chat_history(self, chat_id: str) -> List[Dict]:
        """Retrieve chat history for a given chat ID."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content, metadata, timestamp
                FROM messages
                WHERE chat_id = $1
                ORDER BY timestamp
            """, chat_id)

            return [
                {
                    "role": row['role'],
                    "content": row['content'],
                    "metadata": row['metadata'],
                    "timestamp": row['timestamp']
                }
                for row in rows
            ]

    async def save_indexed_file(
            self,
            file_id: str,
            file_path: str,
            file_type: str,
            metadata: Dict = None,
            embedding_model: str = None,
            chunk_size: int = None,
            chunk_overlap: int = None
    ) -> None:
        """Save information about an indexed file."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO indexed_files 
                (file_id, file_path, file_type, indexed_at, metadata, embedding_model, chunk_size, chunk_overlap)
                VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7)
                ON CONFLICT (file_id) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    file_type = EXCLUDED.file_type,
                    indexed_at = NOW(),
                    metadata = EXCLUDED.metadata,
                    embedding_model = EXCLUDED.embedding_model,
                    chunk_size = EXCLUDED.chunk_size,
                    chunk_overlap = EXCLUDED.chunk_overlap
            """, file_id, file_path, file_type, json.dumps(metadata) if metadata else None,
                               embedding_model, chunk_size, chunk_overlap)

    async def get_indexed_files(self) -> List[Dict]:
        """Retrieve all indexed files."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM indexed_files
                ORDER BY indexed_at DESC
            """)

            return [
                {
                    "file_id": row['file_id'],
                    "file_path": row['file_path'],
                    "file_type": row['file_type'],
                    "indexed_at": row['indexed_at'],
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {},  # Parse JSON string
                    "embedding_model": row['embedding_model'],
                    "chunk_size": row['chunk_size'],
                    "chunk_overlap": row['chunk_overlap']
                }
                for row in rows
            ]

    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and all its messages."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM chats WHERE chat_id = $1", chat_id)

    async def list_chats(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """List all chats with pagination."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chat_id, title, created_at, last_updated
                FROM chats
                ORDER BY last_updated DESC
                LIMIT $1 OFFSET $2
            """, limit, offset)

            return [
                {
                    "chat_id": row['chat_id'],
                    "title": row['title'],
                    "created_at": row['created_at'],
                    "last_updated": row['last_updated']
                }
                for row in rows
            ]

    async def search_messages(
            self,
            query: str,
            chat_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict]:
        """Search messages using full-text search."""
        async with self.pool.acquire() as conn:
            if chat_id:
                rows = await conn.fetch("""
                    SELECT m.*, c.title as chat_title
                    FROM messages m
                    JOIN chats c ON m.chat_id = c.chat_id
                    WHERE m.chat_id = $1 AND m.content ILIKE $2
                    ORDER BY m.timestamp DESC
                    LIMIT $3
                """, chat_id, f"%{query}%", limit)
            else:
                rows = await conn.fetch("""
                    SELECT m.*, c.title as chat_title
                    FROM messages m
                    JOIN chats c ON m.chat_id = c.chat_id
                    WHERE m.content ILIKE $1
                    ORDER BY m.timestamp DESC
                    LIMIT $2
                """, f"%{query}%", limit)

            return [
                {
                    "message_id": row['message_id'],
                    "chat_id": row['chat_id'],
                    "chat_title": row['chat_title'],
                    "role": row['role'],
                    "content": row['content'],
                    "timestamp": row['timestamp'],
                    "metadata": row['metadata']
                }
                for row in rows
            ]