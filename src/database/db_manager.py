from typing import List, Dict, Optional, Any
import asyncpg
import json
from datetime import datetime
import os
import logging
from pathlib import Path
import asyncio
from urllib.parse import urlparse
import traceback
from dataclasses import dataclass


@dataclass
class PoolConfig:
    """Database connection pool configuration."""
    min_size: int = 10
    max_size: int = 20
    max_queries: int = 50000
    timeout: float = 30.0
    command_timeout: float = 60.0
    max_inactive_connection_lifetime: float = 300.0


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""

    def __init__(
            self,
            db_url: Optional[str] = None,
            pool_config: Optional[PoolConfig] = None
    ):
        """
        Initialize DatabaseManager with PostgreSQL connection.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:password@host:port/dbname)
            pool_config: Optional connection pool configuration

        Raises:
            ValueError: If database URL is not provided
        """
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("Database URL must be provided either in constructor or DATABASE_URL environment variable")

        self.pool_config = pool_config or PoolConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        self._init_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self, max_retries: int = 3) -> None:
        """
        Initialize database connection pool and tables.

        Args:
            max_retries: Maximum number of connection retries

        Raises:
            RuntimeError: If initialization fails after retries
        """
        async with self._init_lock:
            if self.pool:
                return

            for attempt in range(max_retries):
                try:
                    self.pool = await asyncpg.create_pool(
                        self.db_url,
                        min_size=self.pool_config.min_size,
                        max_size=self.pool_config.max_size,
                        max_queries=self.pool_config.max_queries,
                        timeout=self.pool_config.timeout,
                        command_timeout=self.pool_config.command_timeout,
                        max_inactive_connection_lifetime=self.pool_config.max_inactive_connection_lifetime
                    )

                    await self._init_db()
                    self._start_cleanup_task()
                    self.logger.info("Database initialized successfully")
                    return

                except Exception as e:
                    self.logger.error(f"Database initialization attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to initialize database after {max_retries} attempts") from e
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def close(self) -> None:
        """Close database connection pool and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.pool:
            await self.pool.close()
            self.pool = None
            self.logger.info("Database connection closed")

    def _start_cleanup_task(self) -> None:
        """Start background task for cleaning up inactive connections."""
        async def cleanup_routine():
            while True:
                try:
                    await asyncio.sleep(600)  # Run every minute
                    if self.pool:
                        await self.pool.execute("SELECT 1")  # Keep-alive query
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup routine error: {str(e)}")

        self._cleanup_task = asyncio.create_task(cleanup_routine())

    async def _init_db(self) -> None:
        """Initialize database tables."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
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

                # Create  indexed_files table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS indexed_files (
                        file_id TEXT PRIMARY KEY,
                        file_path TEXT ,
                        file_type TEXT,
                        indexed_at TIMESTAMP WITH TIME ZONE,
                        metadata JSONB,
                        embedding_model TEXT,
                        chunk_size INTEGER,
                        chunk_overlap INTEGER
                    )
                """)

                # Create indices
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
                    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
                    -- CREATE INDEX IF NOT EXISTS idx_indexed_files_path ON indexed_files(file_path);
                    CREATE INDEX IF NOT EXISTS idx_chats_last_updated ON chats(last_updated);
                """)

    async def _validate_connection(self) -> None:
        """Validate database connection and reinitialize if necessary."""
        if not self.pool:
            await self.initialize()
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
        except (asyncpg.PostgresError, asyncpg.InterfaceError):
            await self.initialize()

    async def create_chat(self, chat_id: str, title: str) -> None:
        """
        Create a new chat session.

        Args:
            chat_id: Unique chat identifier
            title: Chat title
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO chats (chat_id, title, created_at, last_updated)
                    VALUES ($1, $2, NOW(), NOW())
                """, chat_id, title)
        except Exception as e:
            self.logger.error(f"Error creating chat: {str(e)}\n{traceback.format_exc()}")
            raise

    async def save_message(
            self,
            chat_id: str,
            message_id: str,
            role: str,
            content: str,
            metadata: Optional[Dict] = None
    ) -> None:
        """
        Save a message to the database.

        Args:
            chat_id: Chat identifier
            message_id: Unique message identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional message metadata
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("""
                        INSERT INTO messages (message_id, chat_id, role, content, timestamp, metadata)
                        VALUES ($1, $2, $3, $4, NOW(), $5)
                    """, message_id, chat_id, role, content, json.dumps(metadata) if metadata else None)

                    await conn.execute("""
                        UPDATE chats SET last_updated = NOW()
                        WHERE chat_id = $1
                    """, chat_id)
        except Exception as e:
            self.logger.error(f"Error saving message: {str(e)}\n{traceback.format_exc()}")
            raise

    async def get_chat_history(self, chat_id: str, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        """
        Retrieve chat history for a given chat ID.

        Args:
            chat_id: Chat identifier
            limit: Optional limit on number of messages
            offset: Number of messages to skip

        Returns:
            List[Dict]: List of messages with metadata
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT role, content, metadata, timestamp
                    FROM messages
                    WHERE chat_id = $1
                    ORDER BY timestamp
                """
                if limit is not None:
                    query += f" LIMIT {limit} OFFSET {offset}"

                rows = await conn.fetch(query, chat_id)

                return [
                    {
                        "role": row['role'],
                        "content": row['content'],
                        "metadata": row['metadata'],
                        "timestamp": row['timestamp']
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(f"Error getting chat history: {str(e)}\n{traceback.format_exc()}")
            return []

    async def save_indexed_file(
            self,
            file_id: str,
            file_path: str,
            file_type: str,
            metadata: Optional[Dict] = None,
            embedding_model: Optional[str] = None,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None
    ) -> None:
        """Save information about an indexed file."""
        await self._validate_connection()
        try:
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
        except Exception as e:
            self.logger.error(f"Error saving indexed file: {str(e)}\n{traceback.format_exc()}")
            raise

    async def get_indexed_files(self) -> List[Dict]:
        """Retrieve all indexed files."""
        await self._validate_connection()
        try:
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
                        "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                        "embedding_model": row['embedding_model'],
                        "chunk_size": row['chunk_size'],
                        "chunk_overlap": row['chunk_overlap']
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(f"Error getting indexed files: {str(e)}\n{traceback.format_exc()}")
            return []

    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and all its messages."""
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("DELETE FROM chats WHERE chat_id = $1", chat_id)
        except Exception as e:
            self.logger.error(f"Error deleting chat: {str(e)}\n{traceback.format_exc()}")
            raise

    async def list_chats(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """List all chats with pagination."""
        await self._validate_connection()
        try:
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
        except Exception as e:
            self.logger.error(f"Error listing chats: {str(e)}\n{traceback.format_exc()}")
            return []

    async def search_messages(
            self,
            query: str,
            chat_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict]:
        """Search messages using full-text search."""
        await self._validate_connection()
        try:
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
        except Exception as e:
            self.logger.error(f"Error searching messages: {str(e)}\n{traceback.format_exc()}")
            return []

    async def get_chat(self, chat_id: str) -> Optional[Dict]:
        """
        Get chat information by ID.

        Args:
            chat_id: Chat identifier

        Returns:
            Optional[Dict]: Chat information if found
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT chat_id, title, created_at, last_updated
                    FROM chats
                    WHERE chat_id = $1
                """, chat_id)

                if not row:
                    return None

                return {
                    "chat_id": row['chat_id'],
                    "title": row['title'],
                    "created_at": row['created_at'],
                    "last_updated": row['last_updated']
                }
        except Exception as e:
            self.logger.error(f"Error getting chat: {str(e)}\n{traceback.format_exc()}")
            return None

    async def update_chat_title(self, chat_id: str, new_title: str) -> bool:
        """
        Update chat title.

        Args:
            chat_id: Chat identifier
            new_title: New chat title

        Returns:
            bool: True if update was successful
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE chats 
                    SET title = $1, last_updated = NOW()
                    WHERE chat_id = $2
                """, new_title, chat_id)
                return result == "UPDATE 1"
        except Exception as e:
            self.logger.error(f"Error updating chat title: {str(e)}\n{traceback.format_exc()}")
            return False

    async def delete_indexed_file(self, file_id: str) -> bool:
        """
        Delete an indexed file record.

        Args:
            file_id: File identifier

        Returns:
            bool: True if deletion was successful
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    result = await conn.execute("""
                        DELETE FROM indexed_files
                        WHERE file_id = $1
                    """, file_id)
                    return result == "DELETE 1"
        except Exception as e:
            self.logger.error(f"Error deleting indexed file: {str(e)}\n{traceback.format_exc()}")
            return False

    async def get_db_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict[str, Any]: Database statistics
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                stats = {
                    "total_chats": await conn.fetchval("SELECT COUNT(*) FROM chats"),
                    "total_messages": await conn.fetchval("SELECT COUNT(*) FROM messages"),
                    "total_indexed_files": await conn.fetchval("SELECT COUNT(*) FROM indexed_files"),
                    "db_size": await conn.fetchval("""
                        SELECT pg_size_pretty(pg_database_size(current_database()))
                    """),
                    "connection_stats": {
                        "active_connections": len(self.pool._holders) if self.pool else 0,
                        "max_connections": self.pool_config.max_size
                    },
                    "last_vacuum": await conn.fetchval("""
                        SELECT last_vacuum FROM pg_stat_user_tables 
                        WHERE schemaname = 'public' 
                        ORDER BY last_vacuum DESC LIMIT 1
                    """)
                }
                return stats
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}\n{traceback.format_exc()}")
            return {}

    async def get_file_by_path(self, file_path: str) -> Optional[Dict]:
        """
        Get indexed file information by file path.

        Args:
            file_path: Path of the indexed file

        Returns:
            Optional[Dict]: File information if found, None otherwise
        """
        await self._validate_connection()
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT file_id, file_path, file_type, indexed_at, metadata, 
                           embedding_model, chunk_size, chunk_overlap
                    FROM indexed_files
                    WHERE file_path = $1
                """, file_path)

                if not row:
                    return None

                return {
                    "file_id": row['file_id'],
                    "file_path": row['file_path'],
                    "file_type": row['file_type'],
                    "indexed_at": row['indexed_at'],
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                    "embedding_model": row['embedding_model'],
                    "chunk_size": row['chunk_size'],
                    "chunk_overlap": row['chunk_overlap']
                }
        except Exception as e:
            self.logger.error(f"Error getting file by path: {str(e)}\n{traceback.format_exc()}")
            return None