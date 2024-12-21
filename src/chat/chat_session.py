import traceback
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass

from ..models.base_llm import BaseLLM, Message
from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from ..utils.markdown_helper import MarkdownHelper


@dataclass
class ChatContext:
    """Represents the context gathered for a query."""
    content: str
    sources: List[Dict]
    total_tokens: int


class ChatSession:
    def __init__(
            self,
            chat_id: str,
            llm: BaseLLM,
            db_manager: DatabaseManager,
            vector_store: VectorStore,
            embedding_model: BaseEmbeddings,
            title: str = None,
            max_context_tokens: int = 2000,
            max_history_messages: int = 10
    ):
        """
        Initialize a chat session.

        Args:
            chat_id: Unique identifier for the chat
            llm: Language model instance
            db_manager: Database manager instance
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            title: Optional chat title
            max_context_tokens: Maximum tokens for context
            max_history_messages: Maximum number of history messages to include
        """
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

    async def initialize(self) -> None:
        """Initialize chat session in database."""
        try:
            await self.db_manager.create_chat(self.chat_id, self.title)
        except Exception as e:
            self.logger.error(f"Failed to initialize chat session: {str(e)}")
            raise RuntimeError(f"Chat session initialization failed: {str(e)}")

    async def add_message(
            self,
            content: str,
            role: str = "user",
            images: Optional[List[str]] = None,
            documents: Optional[List[str]] = None
    ) -> str:
        """
        Add a message to the chat session.

        Args:
            content: Message content
            role: Message role (user/assistant/system)
            images: Optional list of image paths
            documents: Optional list of document references

        Returns:
            str: Message ID
        """
        try:
            message_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()

            # Validate role
            if role not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {role}")

            # Create message object
            message = Message(
                role=role,
                content=content,
                images=images,
                documents=documents
            )

            # Save to database
            await self.db_manager.save_message(
                chat_id=self.chat_id,
                message_id=message_id,
                role=role,
                content=content,
                metadata={
                    "images": images,
                    "documents": documents,
                    "timestamp": timestamp.isoformat()
                }
            )

            return message_id

        except Exception as e:
            self.logger.error(f"Failed to add message: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to add message: {str(e)}")

    async def get_context(
            self,
            query: str,
            max_chunks: int = 5
    ) -> ChatContext:
        """
        Retrieve relevant context from vector store.

        Args:
            query: The query to find context for
            max_chunks: Maximum number of chunks to retrieve

        Returns:
            ChatContext: Context information including content and sources
        """
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
                content = metadata["text"]
                # Estimate tokens (words * 1.3 for rough estimation)
                estimated_tokens = len(content.split()) * 1.3

                if total_tokens + estimated_tokens > self.max_context_tokens:
                    break

                context_parts.append(content)
                sources.append({
                    "text": content[:200] + "...",  # Preview
                    "source": metadata.get("metadata", {}).get("file_name", "Unknown"),
                    "score": score
                })
                total_tokens += estimated_tokens

            return ChatContext(
                content="\n\n".join(context_parts),
                sources=sources,
                total_tokens=int(total_tokens)
            )

        except Exception as e:
            self.logger.error(f"Failed to get context: {str(e)}\n{traceback.format_exc()}")
            return ChatContext(content="", sources=[], total_tokens=0)

    async def get_chat_history(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            include_metadata: bool = True
    ) -> List[Dict]:
        """
        Retrieve chat history with pagination.

        Args:
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip
            include_metadata: Whether to include message metadata

        Returns:
            List[Dict]: List of messages
        """
        try:
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
            self.logger.error(f"Failed to get chat history: {str(e)}\n{traceback.format_exc()}")
            return []

    async def generate_response(
            self,
            query: str,
            include_images: bool = True,
            include_context: bool = True
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate a response using the LLM.

        Args:
            query: User query
            include_images: Whether to include images in context
            include_context: Whether to include document context

        Returns:
            Tuple[str, Optional[Dict]]: Response text and optional metadata
        """
        try:
            messages = []
            metadata = {}

            # Get relevant context
            if include_context:
                context = await self.get_context(query)
                if context.content:
                    messages.append(Message(
                        role="system",
                        content=f"Use the following relevant context to help answer the question:\n\n{context.content}"
                    ))
                    metadata["context_sources"] = context.sources

            # Add chat history
            history = await self.get_chat_history(limit=self.max_history_messages)
            for msg in history:
                msg_metadata = msg.get("metadata", {})
                messages.append(Message(
                    role=msg["role"],
                    content=msg["content"],
                    images=msg_metadata.get("images") if include_images else None,
                    documents=msg_metadata.get("documents")
                ))

            # Add current query
            messages.append(Message(
                role="user",
                content=query
            ))

            # Generate response
            response = await self.llm.generate_response(messages)

            # Save response
            await self.add_message(
                content=response,
                role="assistant",
                metadata=metadata
            )

            return response, metadata

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def clear_history(self) -> None:
        """Clear chat history."""
        try:
            await self.db_manager.delete_chat(self.chat_id)
        except Exception as e:
            self.logger.error(f"Failed to clear history: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to clear chat history: {str(e)}")