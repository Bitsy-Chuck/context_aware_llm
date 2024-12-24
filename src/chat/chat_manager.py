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

from typing import Dict, List, Optional, Set, Tuple, Any
import uuid
import logging
from datetime import datetime
import traceback
from dataclasses import dataclass
import re

from ..models.base_llm import BaseLLM, Message
from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from ..rag.base_rag import BaseRAG, RAGQuery
from ..rag.factory import RAGFactory, RAGType
from .chat_session import ChatSession, MessageRole, QueryType, ChatContext
from ..rag.graph_rag import GraphRAG


@dataclass
class ChatInfo:
    """Basic information about a chat session."""
    chat_id: str
    title: str
    created_at: datetime
    last_updated: datetime

@dataclass
class CodeContext:
    """Enhanced context for code generation queries."""
    code_snippets: List[str]
    function_signatures: List[str]
    api_endpoints: List[str]
    documentation_sections: List[Dict]
    implementation_examples: List[Dict]
    related_functions: List[str]
    dependencies: List[str]


class ChatManager:
    """Manages multiple chat sessions with enhanced RAG support."""

    def __init__(
            self,
            llm: BaseLLM,
            db_manager: DatabaseManager,
            rag: BaseRAG,
            vector_store: Optional[VectorStore] = None,
            embedding_model: Optional[BaseEmbeddings] = None,
            max_active_sessions: int = 100,
            max_context_tokens: int = 2000,
            rag_type: RAGType = RAGType.GRAPH  # Default to graph RAG for better technical doc handling
    ):
        """
        Initialize ChatManager with RAG support.

        Args:
            llm: Language model instance
            db_manager: Database manager instance
            rag: RAG implementation instance
            vector_store: Optional vector store instance
            embedding_model: Optional embedding model instance
            max_active_sessions: Maximum number of active sessions
            max_context_tokens: Maximum context tokens for RAG
            rag_type: Type of RAG implementation to use
        """
        self.llm = llm
        self.db_manager = db_manager
        self.rag = rag
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.max_active_sessions = max_active_sessions
        self.max_context_tokens = max_context_tokens
        self.rag_type = rag_type
        self.active_sessions: Dict[str, ChatSession] = {}
        self.logger = logging.getLogger(__name__)

        # Code-related patterns
        self.code_keywords = {
            "code", "function", "api", "endpoint", "implementation",
            "example", "snippet", "write", "generate", "usage",
            "documentation", "method", "class", "interface", "chart",
            "visualization", "graph", "plot", "component"
        }

    def _extract_technical_requirements(self, query: str) -> Dict[str, Set[str]]:
        """Extract technical requirements from the query."""
        requirements = {
            "functions": set(),
            "classes": set(),
            "methods": set(),
            "libraries": set(),
            "frameworks": set(),
            "visualizations": set(),
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
                requirements["visualizations"].add(match.lower())
            else:
                requirements["libraries"].add(match.lower())

        # Data types
        data_types = r'\b(string|int|float|bool|array|list|dict|object|number)\b'
        requirements["data_types"].update(re.findall(data_types, query, re.IGNORECASE))

        return requirements

    async def _get_code_context(
            self,
            query: str,
            requirements: Dict[str, Set[str]]
    ) -> Tuple[str, CodeContext]:
        """Get enhanced context for code generation queries."""
        try:
            # Create specialized RAG query
            rag_query = RAGQuery(
                query_text=query,
                filters={"chunk_type": "code"},  # If supported by RAG
                top_k=5  # Adjust based on needs
            )

            # Get results from RAG
            results = await self.rag.search(rag_query)

            code_context = CodeContext(
                code_snippets=[],
                function_signatures=[],
                api_endpoints=[],
                documentation_sections=[],
                implementation_examples=[],
                related_functions=[],
                dependencies=[]
            )

            context_parts = []

            for doc in results.documents:
                context_parts.append(doc.content)

                # Extract code elements if available
                if "code_elements" in doc.metadata:
                    elements = doc.metadata["code_elements"]
                    code_context.code_snippets.extend(elements.get("code_blocks", []))
                    code_context.function_signatures.extend(elements.get("functions", []))
                    code_context.dependencies.extend(elements.get("imports", []))

                # Add documentation sections
                if doc.metadata.get("chunk_type") == "code":
                    code_context.documentation_sections.append({
                        "content": doc.content,
                        "type": "implementation",
                        "score": doc.score
                    })

            # If using GraphRAG, get additional technical context
            if isinstance(self.rag_type, RAGType.GRAPH):
                # Get related function implementations
                for func in requirements["functions"]:
                    implementations = await self.rag.get_implementation_examples(func)
                    code_context.implementation_examples.extend(implementations)

                # Get related endpoints
                endpoints = await self.rag.get_related_endpoints(query)
                code_context.api_endpoints.extend(endpoints)

            return "\n\n".join(context_parts), code_context

        except Exception as e:
            self.logger.error(f"Error getting code context: {str(e)}")
            return "", CodeContext([], [], [], [], [], [], [])

    async def create_chat(self, title: Optional[str] = None) -> ChatSession:
        """Create a new chat session."""
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
                title=title,
                rag=self.rag  # Pass RAG instance
            )

            await session.initialize()
            self.active_sessions[chat_id] = session

            self.logger.info(f"Created new chat session: {chat_id}")
            return session

        except Exception as e:
            error_msg = f"Failed to create chat session: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def generate_response(
            self,
            session: ChatSession,
            query: str,
            include_context: bool = True,
            enable_advanced_context: bool = True
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate a response with enhanced context for code generation."""
        try:
            messages = []
            context_metadata = {}

            # Detect query type and extract requirements
            query_type = session._detect_query_type(query)

            if query_type in [QueryType.CODE, QueryType.VISUALIZATION, QueryType.API] and enable_advanced_context:
                # Extract technical requirements
                requirements = self._extract_technical_requirements(query)

                # Get code-specific context
                context_content, code_context = await self._get_code_context(query, requirements)

                # For complex code generation tasks, get additional context
                enhanced_context = {}
                try:
                    if isinstance(self.rag, GraphRAG):
                        # Get implementation examples for required functions
                        impl_examples = []
                        for func in requirements["functions"]:
                            examples = await self.rag.get_implementation_examples(func)
                            impl_examples.extend(examples)

                        # Get related API endpoints
                        related_endpoints = []
                        for func in requirements["functions"]:
                            endpoints = await self.rag.get_related_endpoints(func)
                            related_endpoints.extend(endpoints)

                        # Analyze dependencies
                        dep_analysis = await self.rag.analyze_dependencies()

                        # Get document hierarchy for complex components
                        if query_type == QueryType.VISUALIZATION:
                            doc_hierarchy = await self.rag.get_document_hierarchy()

                            # Find relevant component implementations
                            viz_components = [
                                doc for doc in doc_hierarchy["implementation_groups"]
                                if any(viz in doc.lower() for viz in requirements["visualizations"])
                            ]

                        # Enhanced context construction
                        enhanced_context = {
                            "implementation_examples": impl_examples,
                            "related_endpoints": related_endpoints,
                            "dependency_analysis": {
                                "central_dependencies": dep_analysis["central_dependencies"],
                                "dependency_clusters": dep_analysis["dependency_clusters"]
                            }
                        }

                        if query_type == QueryType.VISUALIZATION:
                            enhanced_context["visualization_components"] = viz_components

                        # Update metadata
                        context_metadata.update({
                            "enhanced_context": enhanced_context,
                            "implementation_examples": len(impl_examples),
                            "related_endpoints": len(related_endpoints),
                            "analyzed_dependencies": len(dep_analysis["central_dependencies"])
                        })
                    else:
                        # If not using GraphRAG, still provide basic code context
                        context_metadata["rag_type"] = self.rag.get_backend_type()
                        self.logger.info(
                            f"Using basic code context with {self.rag.get_backend_type()} RAG implementation")

                except Exception as e:
                    self.logger.error(f"Error getting enhanced context: {str(e)}")
                    # Continue with basic context even if enhanced context fails
                    context_metadata["enhanced_context_error"] = str(e)

                if context_content:
                    context_prompt = (
                        "Use the following technical documentation and code examples to help generate the code:\n\n"
                        f"{context_content}\n\n"
                    )

                    if code_context.implementation_examples:
                        context_prompt += "\nRelevant Implementation Examples:\n"
                        for example in code_context.implementation_examples:
                            context_prompt += f"\n{example}\n"

                    messages.append(Message(
                        role=MessageRole.SYSTEM.value,
                        content=context_prompt
                    ))

                    context_metadata = {
                        "query_type": query_type,
                        "technical_requirements": requirements,
                        "code_snippets": len(code_context.code_snippets),
                        "function_signatures": len(code_context.function_signatures),
                        "api_endpoints": len(code_context.api_endpoints),
                        "implementation_examples": len(code_context.implementation_examples)
                    }

            else:
                # Use standard context retrieval for non-code queries
                if include_context:
                    context = await session.get_context(query)
                    if context.content:
                        messages.append(Message(
                            role=MessageRole.SYSTEM.value,
                            content=f"Use this context to help answer:\n\n{context.content}"
                        ))
                        context_metadata = {
                            "query_type": QueryType.GENERAL,
                            "context_length": len(context.content)
                        }

            # Add chat history
            history = await session.get_chat_history()
            messages.extend([
                Message(role=msg["role"], content=msg["content"])
                for msg in history
            ])

            # Add current query
            messages.append(Message(
                role=MessageRole.USER.value,
                content=query
            ))

            # Generate response
            response = await self.llm.generate_response(messages)

            # Save message
            await session.add_message(
                content=response,
                role=MessageRole.ASSISTANT.value
            )

            return response, context_metadata

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
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