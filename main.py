import asyncio
import logging
import os
from pathlib import Path
import argparse
import sys
from typing import Tuple, Optional
import traceback

from bridge import WebUIBridge
from src.models.chat_gpt import ChatGPTModel
from src.models.claude_model import ClaudeModel
from src.models.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.database.db_manager import DatabaseManager
from src.database.vector_store import VectorStore
from src.chat.chat_manager import ChatManager
from src.chat.message_formatter import MessageFormatter
from src.indexing.index_manager import IndexManager
from src.ui.terminal_ui import TerminalUI
from src.utils.config import ConfigManager
from src.rag.factory import RAGFactory, RAGType


class ApplicationContext:
    """Manages application-wide components and their lifecycle."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager: Optional[DatabaseManager] = None
        self.chat_manager: Optional[ChatManager] = None
        self.index_manager: Optional[IndexManager] = None
        self.message_formatter: Optional[MessageFormatter] = None
        self.rag_system = None

    async def initialize(self, config_path: str) -> Tuple[ChatManager, IndexManager, MessageFormatter]:
        """
        Initialize all application components.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple containing initialized managers

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Load configuration
            config_manager = ConfigManager(config_path)
            config = config_manager.get_config()

            # Initialize database
            self.db_manager = DatabaseManager(config.database.db_url)
            await self.db_manager.initialize()

            # Initialize embedding model
            embedding_model = SentenceTransformerEmbeddings(
                model_name=config.embedding.model_name,
                batch_size=config.embedding.batch_size
            )

            # Initialize vector store
            vector_store = VectorStore(
                dimension=embedding_model.get_embedding_dimension(),
                index_path=config.vector_store_path
            )

            # Initialize RAG system
            rag_type = getattr(config, 'rag_type', 'vector')
            rag_factory = RAGFactory()
            self.rag_system = await rag_factory.create_rag(
                rag_type=RAGType.VECTOR,  #TODO: replace with enum impl
                embedding_model=embedding_model,
                db_manager=self.db_manager,
                vector_store=vector_store
            )

            # Initialize LLM based on config
            if getattr(config.model, 'type', 'gpt').lower() == 'claude':
                llm = ClaudeModel(
                    model_name=config.model.model_name,
                    api_key=config.model.api_key,
                    max_tokens=config.model.max_tokens,
                )
            else:
                llm = ChatGPTModel(
                    api_key=config.model.api_key,
                    api_version=config.model.api_version,
                    api_base=config.model.api_base,

                )

            # Initialize managers with RAG support
            self.chat_manager = ChatManager(
                llm=llm,
                db_manager=self.db_manager,
                vector_store=vector_store,
                embedding_model=embedding_model,
                rag=self.rag_system
            )

            self.index_manager = IndexManager(
                embedding_model=embedding_model,
                db_manager=self.db_manager,
                vector_store=vector_store,
                rag_type=RAGType.VECTOR,
            )

            self.message_formatter = MessageFormatter()

            return self.chat_manager, self.index_manager, self.message_formatter

        except Exception as e:
            error_msg = f"Failed to initialize application components: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    async def cleanup(self) -> None:
        """Cleanup and close all components."""
        try:
            if self.db_manager:
                await self.db_manager.close()
            if self.rag_system and hasattr(self.rag_system, 'cleanup'):
                await self.rag_system.cleanup()
            self.logger.info("Successfully cleaned up application components")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}\n{traceback.format_exc()}")


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
        log_dir: Directory for log files
    """
    try:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, 'app.log'))
            ]
        )
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}", file=sys.stderr)
        raise


def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG-powered Chat System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )

    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files"
    )

    parser.add_argument(
        "--rag-type",
        default=None,
        choices=["vector", "graph"],
        help="Override RAG type from config file"
    )

    return parser


async def main() -> None:
    """Main application entry point."""
    global logger
    app_context = ApplicationContext()

    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Create required directories
        os.makedirs("config", exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        # Setup logging
        setup_logging(args.log_level, args.log_dir)
        logger = logging.getLogger(__name__)
        logger.info("Starting application...")

        # Initialize components
        chat_manager, index_manager, message_formatter = await app_context.initialize(args.config)

        # # Create and start UI
        # ui = TerminalUI(chat_manager, index_manager, message_formatter)
        # logger.info("Application initialized successfully")
        # Start UI
        # await ui.start()

        bridge = WebUIBridge(
            chat_manager=chat_manager,
            index_manager=index_manager,
            message_formatter=message_formatter,
            host="localhost",
            port=8000
        )

        try:
            await bridge.start()
        except KeyboardInterrupt:
            await bridge.stop()

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Cleanup
        await app_context.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
