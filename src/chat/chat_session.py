from typing import List, Dict, Optional, Union, Tuple, Set
from datetime import datetime
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import traceback
import re

from ..models.base_llm import BaseLLM, Message
from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from ..rag.base_rag import BaseRAG, RAGQuery
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
    code_snippets: Optional[List[str]] = None
    function_signatures: Optional[List[str]] = None
    api_endpoints: Optional[List[str]] = None


@dataclass
class ChatContext:
    """Represents the context gathered for a query."""
    content: str
    sources: List[Source]
    total_tokens: int
    is_code_query: bool = False
    code_snippets: List[str] = None
    function_signatures: List[str] = None
    api_endpoints: List[str] = None
    documentation_sections: List[Dict] = None

    def __post_init__(self):
        """Initialize optional fields and validate the context data."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be a string")
        if not isinstance(self.sources, list):
            raise ValueError("Sources must be a list")
        if not isinstance(self.total_tokens, int) or self.total_tokens < 0:
            raise ValueError("Total tokens must be a non-negative integer")

        self.code_snippets = self.code_snippets or []
        self.function_signatures = self.function_signatures or []
        self.api_endpoints = self.api_endpoints or []
        self.documentation_sections = self.documentation_sections or []


class QueryType(str, Enum):
    """Types of queries that can be handled."""
    CODE = "code"
    GENERAL = "general"
    API = "api"
    VISUALIZATION = "visualization"


class ChatSession:
    def __init__(
            self,
            chat_id: str,
            llm: BaseLLM,
            db_manager: DatabaseManager,
            rag: BaseRAG,
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
        self.rag = rag
        self.embedding_model = embedding_model
        self.title = title or f"Chat {chat_id[:8]}"
        self.max_context_tokens = max_context_tokens
        self.max_history_messages = max_history_messages
        self.markdown_helper = MarkdownHelper()
        self.logger = logging.getLogger(__name__)
        self.last_accessed = datetime.utcnow()

        # Code-related patterns
        self.code_keywords = {
            "code", "function", "api", "endpoint", "implementation",
            "example", "snippet", "write", "generate", "usage",
            "documentation", "method", "class", "interface", "chart",
            "visualization", "graph", "plot", "component"
        }

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query being made."""
        query_lower = query.lower()

        # Check for visualization-related queries
        viz_keywords = {"chart", "plot", "graph", "visualization", "dashboard"}
        if any(keyword in query_lower for keyword in viz_keywords):
            return QueryType.VISUALIZATION

        # Check for API-related queries
        api_keywords = {"api", "endpoint", "route", "request", "response"}
        if any(keyword in query_lower for keyword in api_keywords):
            return QueryType.API

        # Check for general code queries
        if any(keyword in query_lower for keyword in self.code_keywords):
            return QueryType.CODE

        return QueryType.GENERAL

    def _extract_technical_requirements(self, query: str) -> Dict[str, Set[str]]:
        """Extract technical requirements from the query."""
        requirements = {
            "functions": set(),
            "classes": set(),
            "methods": set(),
            "libraries": set(),
            "frameworks": set(),
            "data_types": set()
        }

        # Extract function names
        func_pattern = r'\b(?:function|method|def)\s+([a-zA-Z_]\w*)'
        requirements["functions"].update(re.findall(func_pattern, query, re.IGNORECASE))

        # Extract class names
        class_pattern = r'\b(?:class)\s+([a-zA-Z_]\w*)'
        requirements["classes"].update(re.findall(class_pattern, query, re.IGNORECASE))

        # Extract method references
        method_pattern = r'\.([a-zA-Z_]\w*)\('
        requirements["methods"].update(re.findall(method_pattern, query))

        # Common libraries and frameworks
        tech_keywords = r'\b(react|vue|angular|tensorflow|pytorch|pandas|numpy|matplotlib|seaborn|plotly|d3|recharts)\b'
        matches = re.findall(tech_keywords, query, re.IGNORECASE)
        for match in matches:
            if any(viz_lib in match.lower() for viz_lib in ['matplotlib', 'seaborn', 'plotly', 'd3', 'recharts']):
                requirements["frameworks"].add(match.lower())
            else:
                requirements["libraries"].add(match.lower())

        # Data types
        data_types = r'\b(string|int|float|bool|array|list|dict|object|number)\b'
        requirements["data_types"].update(re.findall(data_types, query, re.IGNORECASE))

        return requirements

    async def _get_code_context(self, query: str, requirements: Dict[str, Set[str]],
                                max_chunks: int = 8) -> ChatContext:
        """Get context specifically for code-related queries."""
        try:
            # Create RAG query with code-specific parameters
            rag_query = RAGQuery(
                query_text=query,
                top_k=max_chunks,
                filters={"chunk_type": "code"}  # If supported by RAG implementation
            )

            # Get results from RAG
            results = await self.rag.search(rag_query)

            context_parts = []
            sources = []
            total_tokens = 0

            # Collect code-specific elements
            code_snippets = []
            function_signatures = []
            api_endpoints = []
            documentation_sections = []

            for doc in results.documents:
                estimated_tokens = len(doc.content.encode('utf-8')) // 4

                if total_tokens + estimated_tokens > self.max_context_tokens:
                    break

                # Extract code-specific content
                if hasattr(doc.metadata, "code_context"):
                    code_context = doc.metadata.code_context
                    code_snippets.extend(code_context.get("code_blocks", []))
                    function_signatures.extend(code_context.get("related_functions", []))
                    documentation_sections.append({
                        "content": doc.content,
                        "type": "implementation" if code_snippets else "documentation",
                        "score": doc.score
                    })

                source = Source(
                    text=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    file_name=doc.metadata.get("source_file", "Unknown"),
                    score=doc.score if doc.score is not None else 0.0,
                    chunk_index=doc.metadata.get("chunk_index"),
                    code_snippets=code_snippets,
                    function_signatures=function_signatures,
                    api_endpoints=api_endpoints
                )

                context_parts.append(doc.content)
                sources.append(source)
                total_tokens += estimated_tokens

            return ChatContext(
                content="\n\n".join(context_parts),
                sources=sources,
                total_tokens=total_tokens,
                is_code_query=True,
                code_snippets=code_snippets,
                function_signatures=function_signatures,
                api_endpoints=api_endpoints,
                documentation_sections=documentation_sections
            )

        except Exception as e:
            self.logger.error(f"Error getting code context: {str(e)}")
            return ChatContext(content="", sources=[], total_tokens=0, is_code_query=True)

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

            # Detect query type and extract requirements
            query_type = self._detect_query_type(query)
            technical_requirements = None
            if self.rag.get_backend_type() == "vector":
                query_type = QueryType.GENERAL

            if query_type in [QueryType.CODE, QueryType.VISUALIZATION, QueryType.API]:
                technical_requirements = self._extract_technical_requirements(query)

            # Get relevant context based on query type
            if include_context:
                if query_type in [QueryType.CODE, QueryType.VISUALIZATION, QueryType.API]:
                    context = await self._get_code_context(query, technical_requirements)

                    # Prepare specialized context prompt
                    context_prompt = "Use the following technical documentation and code examples to help generate the response:\n\n"

                    if context.documentation_sections:
                        context_prompt += "\nRelevant Documentation:\n"
                        for section in context.documentation_sections:
                            context_prompt += f"\n{section['content']}\n"

                    if context.code_snippets:
                        context_prompt += "\nRelevant Code Examples:\n"
                        for snippet in context.code_snippets:
                            context_prompt += f"\n```\n{snippet}\n```\n"

                    messages.append(Message(
                        role=MessageRole.SYSTEM.value,
                        content=context_prompt
                    ))

                    context_metadata = {
                        "query_type": query_type,
                        "technical_requirements": technical_requirements,
                        "sources": [vars(source) for source in context.sources],
                        "total_tokens": context.total_tokens,
                        "code_snippets": len(context.code_snippets),
                        "function_signatures": len(context.function_signatures),
                        "documentation_sections": len(context.documentation_sections)
                    }
                else:
                    # Use standard context retrieval for non-code queries
                    context = await self.get_context(query)
                    if context.content:
                        messages.append(Message(
                            role=MessageRole.SYSTEM.value,
                            content=f"Use the following relevant context to help answer the question:\n\n{context.content}"
                        ))
                        context_metadata = {
                            "query_type": query_type,
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

            self.logger.info(f"Generating response for chat {self.chat_id} with {len(messages)} messages")

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

    # [Previous methods remain unchanged]
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
                metadata={
                    "has_images": bool(images),
                    "has_documents": bool(documents),
                    "images": images,
                    "documents": documents,
                    "timestamp": datetime.utcnow().isoformat()
                }
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
            threshold: float = 0.3
    ) -> ChatContext:
        """Retrieve relevant context from vector store."""
        try:
            # Create RAG query
            rag_query = RAGQuery(
                query_text=query,
                top_k=max_chunks
            )

            # Get results from RAG
            results = await self.rag.search(rag_query)

            context_parts = []
            sources = []
            total_tokens = 0

            for doc in results.documents:
                # Skip if below relevance threshold
                if doc.score and doc.score < threshold:
                    continue

                # Improved token estimation
                estimated_tokens = len(doc.content.encode('utf-8')) // 4

                if total_tokens + estimated_tokens > self.max_context_tokens:
                    break

                context_parts.append(doc.content)
                sources.append(Source(
                    text=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    file_name=doc.metadata.get("source_file", "Unknown"),
                    score=float(doc.score) if doc.score is not None else 0.0,
                    chunk_index=doc.metadata.get("chunk_index")
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

    async def clear_history(self) -> None:
        """Clear chat history."""
        try:
            await self.db_manager.delete_chat(self.chat_id)
            self.logger.info(f"Cleared history for chat {self.chat_id}")
        except Exception as e:
            error_msg = f"Failed to clear chat history: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)