```markdown
# RAG-Powered Chat System

A terminal-based chat system that combines the power of Large Language Models with RAG (Retrieval Augmented Generation) capabilities. This system allows for multiple chat sessions, document indexing, and context-aware responses.

## Features

- 💬 **Multiple Chat Sessions**: Create and manage multiple independent chat conversations
- 📄 **Document Integration**: Upload and index various document types for context
- 🔍 **RAG-Powered Responses**: All responses are enhanced with relevant context from indexed documents
- 🎨 **Rich Formatting**: Markdown support with syntax highlighting for code blocks
- 🔌 **Pluggable Architecture**: Easily swap LLM and embedding models
- 💾 **Persistent Storage**: All chats and indexed documents are saved to PostgreSQL
- 📱 **Terminal UI**: Clean and intuitive terminal-based user interface
- 🖼️ **Image Support**: Include images in your conversations
- ⚡ **Fast Search**: Vector-based similarity search for relevant context

## Installation

1. Clone the repository:
```bash
https://github.com/Bitsy-Chuck/context_aware_llm.git
cd context_aware_llm
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL database and set environment variables:
```bash
# Create database
createdb rag_chat

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key"
export DATABASE_URL="postgresql://user:password@localhost:5432/rag_chat"
```

## Configuration

The system uses a YAML configuration file located at `config/default_config.yaml`. You can customize:

- LLM model settings
- Embedding model parameters
- Database configuration
- Chunking and RAG parameters
- UI preferences
- Logging settings

## Usage

Start the application:
```bash
python main.py
```

Or with custom configuration:
```bash
python main.py --config path/to/config.yaml --log-level DEBUG
```

### Available Commands

- `/new` - Create a new chat session
- `/switch <chat_id>` - Switch to another chat
- `/list` - List all chats
- `/upload` - Upload and index a file
- `/files` - List indexed files
- `/clear` - Clear the screen
- `/help` - Show help information
- `/exit` - Exit application

### Code Structure

```
rag_chat_system/
├── src/
│   ├── models/          # LLM and embedding model implementations
│   ├── database/        # Database and vector store management
│   ├── chat/           # Chat session and message handling
│   ├── indexing/       # Document processing and indexing
│   ├── utils/          # Utility functions and helpers
│   └── ui/             # Terminal UI implementation
├── config/             # Configuration files
└── main.py            # Application entry point
```

## Dependencies

Major dependencies include:
- `anthropic` - Claude API client
- `sentence-transformers` - For document embeddings
- `asyncpg` - PostgreSQL async driver
- `faiss-cpu` - Vector similarity search
- `rich` - Terminal UI formatting
- `prompt_toolkit` - Interactive terminal interface
- `PyYAML` - Configuration management

## Development

### Adding a New LLM

1. Create a new class in `models/` that inherits from `BaseLLM`
2. Implement the required methods
3. Update the configuration to use your new model

### Adding New Document Types

1. Update `FileHelper` in `utils/file_helper.py`
2. Add processing logic in `indexing/document_processor.py`
3. Update supported file types in configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic](https://www.anthropic.com/) for the Claude API
- [sentence-transformers](https://www.sbert.net/) for embedding capabilities
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Rich](https://rich.readthedocs.io/) for terminal formatting

## Support

For support:
1. Check the `/help` command in the application
2. Open an issue on GitHub
3. Contact the maintenance team

## Requirements

- Python 3.8+
- PostgreSQL 12+
- 8GB RAM minimum (recommended)
- Anthropic API key
```
