from typing import Dict, List, Optional
import uuid
import logging
from datetime import datetime
import traceback
from dataclasses import dataclass

from ..models.base_llm import BaseLLM
from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from .chat_session import ChatSession


@dataclass
class ChatInfo:
    """Basic information about a chat session."""
    chat_id: str
    title: str
    created_at: datetime
    last_updated: datetime


class ChatManager:
    """Manages multiple chat sessions and their lifecycle."""

    def __init__(
            self,
            llm: BaseLLM,
            db_manager: DatabaseManager,
            vector_store: VectorStore,
            embedding_model: BaseEmbeddings,
            max_active_sessions: int = 100
    ):
        """
        Initialize ChatManager.

        Args:
            llm: Language model instance
            db_manager: Database manager instance
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            max_active_sessions: Maximum number of active sessions to keep in memory
        """
        self.llm = llm
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.max_active_sessions = max_active_sessions
        self.active_sessions: Dict[str, ChatSession] = {}
        self.logger = logging.getLogger(__name__)

    async def create_chat(self, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.

        Args:
            title: Optional title for the chat session

        Returns:
            ChatSession: Newly created chat session

        Raises:
            RuntimeError: If session creation fails
        """
        try:
            chat_id = str(uuid.uuid4())

            # Clean up old sessions if needed
            if len(self.active_sessions) >= self.max_active_sessions:
                oldest_chat_id = min(
                    self.active_sessions.keys(),
                    key=lambda x: self.active_sessions[x].last_accessed
                )
                self.active_sessions.pop(oldest_chat_id, None)

            # Create new session
            session = ChatSession(
                chat_id=chat_id,
                llm=self.llm,
                db_manager=self.db_manager,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                title=title
            )

            await session.initialize()
            self.active_sessions[chat_id] = session

            self.logger.info(f"Created new chat session: {chat_id}")
            return session

        except Exception as e:
            error_msg = f"Failed to create chat session: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def get_chat(self, chat_id: str) -> Optional[ChatSession]:
        """
        Get an existing chat session.

        Args:
            chat_id: ID of the chat session to retrieve

        Returns:
            Optional[ChatSession]: The chat session if found, None otherwise
        """
        try:
            # Return from active sessions if available
            if chat_id in self.active_sessions:
                return self.active_sessions[chat_id]

            # Try to load from database
            chat_info = await self.db_manager.get_chat(chat_id)
            if not chat_info:
                self.logger.debug(f"Chat session not found: {chat_id}")
                return None

            # Create new session instance
            session = ChatSession(
                chat_id=chat_id,
                llm=self.llm,
                db_manager=self.db_manager,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                title=chat_info["title"]
            )

            # Add to active sessions
            self.active_sessions[chat_id] = session
            self.logger.debug(f"Loaded chat session: {chat_id}")
            return session

        except Exception as e:
            self.logger.error(f"Error retrieving chat {chat_id}: {str(e)}\n{traceback.format_exc()}")
            return None

    async def list_chats(
            self,
            limit: int = 10,
            offset: int = 0
    ) -> List[ChatInfo]:
        """
        List all chat sessions with pagination.

        Args:
            limit: Maximum number of chats to return
            offset: Number of chats to skip

        Returns:
            List[ChatInfo]: List of chat information
        """
        try:
            chats = await self.db_manager.list_chats(limit, offset)
            return [
                ChatInfo(
                    chat_id=chat["chat_id"],
                    title=chat["title"],
                    created_at=chat["created_at"],
                    last_updated=chat["last_updated"]
                )
                for chat in chats
            ]
        except Exception as e:
            self.logger.error(f"Error listing chats: {str(e)}\n{traceback.format_exc()}")
            return []

    async def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat session.

        Args:
            chat_id: ID of the chat session to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Remove from active sessions
            self.active_sessions.pop(chat_id, None)

            # Delete from database
            await self.db_manager.delete_chat(chat_id)

            self.logger.info(f"Deleted chat session: {chat_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting chat {chat_id}: {str(e)}\n{traceback.format_exc()}")
            return False

    async def search_messages(
            self,
            query: str,
            chat_id: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict]:
        """
        Search messages across all chats or within a specific chat.

        Args:
            query: Search query string
            chat_id: Optional chat ID to limit search scope
            limit: Maximum number of messages to return

        Returns:
            List[Dict]: List of matching messages with metadata
        """
        try:
            return await self.db_manager.search_messages(query, chat_id, limit)
        except Exception as e:
            self.logger.error(f"Error searching messages: {str(e)}\n{traceback.format_exc()}")
            return []

    async def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> None:
        """
        Clean up inactive chat sessions from memory.

        Args:
            max_age_hours: Maximum age of inactive sessions in hours
        """
        try:
            current_time = datetime.utcnow()
            sessions_to_remove = [
                chat_id for chat_id, session in self.active_sessions.items()
                if (current_time - session.last_accessed).total_seconds() > max_age_hours * 3600
            ]

            for chat_id in sessions_to_remove:
                self.active_sessions.pop(chat_id, None)
                self.logger.debug(f"Cleaned up inactive session: {chat_id}")

        except Exception as e:
            self.logger.error(f"Error during session cleanup: {str(e)}\n{traceback.format_exc()}")