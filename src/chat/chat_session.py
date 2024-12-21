from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import traceback

from ..models.base_llm import BaseLLM, Message
from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from ..utils.markdown_helper import MarkdownHelper


class MessageRole(str, Enum):
    """Valid message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Source:
    """Information about a context source."""
    text: str
    file_name: str
    score: float
    chunk_index: Optional[int] = None


@dataclass
class ChatContext:
    """Represents the context gathered for a query."""
    content: str
    sources: List[Source]
    total_tokens: int

    def __post_init__(self):
        """Validate the context data."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be a string")
        if not isinstance(self.sources, list):
            raise ValueError("Sources must be a list")
        if not isinstance(self.total_tokens, int) or self.total_tokens < 0:
            raise ValueError("Total tokens must be a non-negative integer")


class ChatSession:
    def __init__(
            self,
            chat_id: str,
            llm: BaseLLM,
            db_manager: DatabaseManager,
            vector_store: VectorStore,
            embedding_model: BaseEmbeddings,
            title: Optional[str] = None,
            max_context_tokens: int = 2000,
            max_history_messages: int = 10
    ):
        """Initialize a chat session."""
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be positive")

        self.chat_id = chat_id
        self.llm = llm
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.title = title or f"Chat {chat_id[:8]}"
        self.max_context_tokens = max_context_tokens
        self.max_history_messages = max_history_messages
        self.markdown_helper = MarkdownHelper()
        self.logger = logging.getLogger(__name__)
        self.last_accessed = datetime.utcnow()

    async def initialize(self) -> None:
        """Initialize chat session in database."""
        try:
            await self.db_manager.create_chat(self.chat_id, self.title)
            self.logger.info(f"Initialized chat session: {self.chat_id}")
        except Exception as e:
            error_msg = f"Failed to initialize chat session: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def add_message(
            self,
            content: str,
            role: Union[MessageRole, str] = MessageRole.USER,
            images: Optional[List[str]] = None,
            documents: Optional[List[str]] = None
    ) -> str:
        """Add a message to the chat session."""
        try:
            # Update last accessed timestamp
            self.last_accessed = datetime.utcnow()

            # Validate content
            if not content or not content.strip():
                raise ValueError("Message content cannot be empty")
            if len(content) > 32768:  # 32KB limit
                raise ValueError("Message content too large")

            # Convert role string to enum if needed
            if isinstance(role, str):
                try:
                    role = MessageRole(role)
                except ValueError:
                    raise ValueError(f"Invalid role: {role}")

            # Generate message ID
            message_id = str(uuid.uuid4())

            # Save to database
            await self.db_manager.save_message(
                chat_id=self.chat_id,
                message_id=message_id,
                role=str(role.value),
                content=content,
                # images=images,
                # documents=documents
            )

            self.logger.debug(f"Added message {message_id} to chat {self.chat_id}")
            return message_id

        except Exception as e:
            error_msg = f"Failed to add message: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def get_context(
            self,
            query: str,
            max_chunks: int = 5,
            threshold: float = 0.6
    ) -> ChatContext:
        """Retrieve relevant context from vector store."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.embed_query(query)

            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=max_chunks,
                filter_criteria=None
            )

            context_parts = []
            sources = []
            total_tokens = 0

            for _, score, metadata in results:
                # Skip if below relevance threshold
                if score < threshold:
                    continue

                content = metadata["text"]
                # Improved token estimation
                estimated_tokens = len(content.encode('utf-8')) // 4  # Better approximation

                if total_tokens + estimated_tokens > self.max_context_tokens:
                    break

                context_parts.append(content)
                sources.append(Source(
                    text=content[:200] + "..." if len(content) > 200 else content,
                    file_name=metadata.get("metadata", {}).get("file_name", "Unknown"),
                    score=float(score),
                    chunk_index=metadata.get("metadata", {}).get("chunk_index")
                ))
                total_tokens += estimated_tokens

            return ChatContext(
                content="\n\n".join(context_parts),
                sources=sources,
                total_tokens=total_tokens
            )

        except Exception as e:
            error_msg = f"Failed to get context: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return ChatContext(content="", sources=[], total_tokens=0)

    async def get_chat_history(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            include_metadata: bool = True
    ) -> List[Dict]:
        """Retrieve chat history with pagination."""
        try:
            self.last_accessed = datetime.utcnow()

            messages = await self.db_manager.get_chat_history(
                self.chat_id,
                limit=limit or self.max_history_messages,
                offset=offset
            )

            if not include_metadata:
                for msg in messages:
                    msg.pop("metadata", None)

            return messages

        except Exception as e:
            error_msg = f"Failed to get chat history: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return []

    async def generate_response(
            self,
            query: str,
            include_images: bool = True,
            include_context: bool = True
    ) -> Tuple[str, Optional[Dict]]:
        """Generate a response using the LLM."""
        try:
            self.last_accessed = datetime.utcnow()
            messages = []
            context_metadata = {}

            # Get relevant context
            if include_context:
                context = await self.get_context(query)
                if context.content:
                    messages.append(Message(
                        role=MessageRole.SYSTEM.value,
                        content=f"Use the following relevant context to help answer the question:\n\n{context.content}"
                    ))
                    context_metadata = {
                        "sources": [vars(source) for source in context.sources],
                        "total_tokens": context.total_tokens
                    }

            # Add chat history
            history = await self.get_chat_history(limit=self.max_history_messages)
            for msg in history:
                messages.append(Message(
                    role=msg["role"],
                    content=msg["content"],
                    images=msg.get("images") if include_images else None,
                    documents=msg.get("documents")
                ))

            # Add current query
            messages.append(Message(
                role=MessageRole.USER.value,
                content=query
            ))

            self.logger.info(f"Generating response for chat {self.chat_id} with {len(messages)} messages", messages)

            # Generate response
            response = await self.llm.generate_response(messages)

            # Save response
            await self.add_message(
                content=response,
                role=MessageRole.ASSISTANT
            )

            return response, context_metadata

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def clear_history(self) -> None:
        """Clear chat history."""
        try:
            await self.db_manager.delete_chat(self.chat_id)
            self.logger.info(f"Cleared history for chat {self.chat_id}")
        except Exception as e:
            error_msg = f"Failed to clear chat history: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)