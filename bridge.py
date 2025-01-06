import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

class WebSocketConnection:
    """Represent a single WebSocket connection."""

    def __init__(self, websocket: WebSocket, connected_at: datetime):
        self.websocket = websocket
        self.connected_at = connected_at
        self.last_ping = datetime.now()
        self.user_id: Optional[str] = None
        self.metadata: Dict[str, any] = {}

    async def ping(self):
        """Send a ping message and update last ping time."""
        try:
            await self.websocket.send_json({"type": "ping"})
            self.last_ping = datetime.now()
            return True
        except Exception:
            return False

    def is_stale(self, timeout_seconds: int) -> bool:
        """Check if connection is stale based on last ping."""
        return (datetime.now() - self.last_ping).total_seconds() > timeout_seconds
# Fix 1: Move Config class definition to top
class Config:
    """Configuration class for the WebUI Bridge."""

    def __init__(self):
        self.max_connections = 100
        self.ping_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        self.max_message_size = 1024 * 1024  # 1MB
        self.allowed_origins = ["*"]  # CORS origins
        self.debug_mode = False


# Fix 2: Add ConnectionManager and WebSocketConnection classes
class ConnectionManager:
    """Manage WebSocket connections and their state."""

    def __init__(self, config: Config):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.config = config
        self._lock = asyncio.Lock()

    async def add_connection(self, websocket: WebSocket) -> str:
        """Add a new WebSocket connection."""
        async with self._lock:
            if len(self.connections) >= self.config.max_connections:
                raise Exception("Maximum connections reached")

            connection_id = str(uuid.uuid4())
            self.connections[connection_id] = WebSocketConnection(
                websocket=websocket,
                connected_at=datetime.now()
            )
            return connection_id

    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if connection_id in self.connections:
                conn = self.connections.pop(connection_id)
                await conn.websocket.close()

    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        """Broadcast a message to all connections except excluded one."""
        failed_connections = []

        for conn_id, conn in self.connections.items():
            if conn_id != exclude:
                try:
                    await conn.websocket.send_json(message)
                except Exception:
                    failed_connections.append(conn_id)

        # Clean up failed connections
        for conn_id in failed_connections:
            await self.remove_connection(conn_id)

    async def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get a connection by ID."""
        return self.connections.get(connection_id)

    async def ping_all(self):
        """Ping all connections to check their status."""
        failed_connections = []

        for conn_id, conn in self.connections.items():
            if not await conn.ping():
                failed_connections.append(conn_id)

        # Clean up failed connections
        for conn_id in failed_connections:
            await self.remove_connection(conn_id)



# Fix 3: Update WebUIBridge class with missing methods and proper class structure
class WebUIBridge:
    def __init__(
            self,
            chat_manager,
            index_manager,
            message_formatter,
            host: str = "localhost",
            port: int = 8000
    ):
        # Initialize basic attributes
        self.chat_manager = chat_manager
        self.index_manager = index_manager
        self.message_formatter = message_formatter
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self._start_time = datetime.now()

        # Initialize config first
        self.config = Config()

        # Initialize FastAPI app
        self.app = FastAPI()

        # Initialize managers after config is created
        self.connection_manager = ConnectionManager(self.config)
        self.metrics = WebUIMetrics()

        # Setup application
        self.setup_cors()
        self.setup_routes()
        self.configure_endpoints()

    # def setup_cors(self):
    #     """Configure CORS for the application."""
    #     self.app.add_middleware(
    #         CORSMiddleware,
    #         allow_origins=self.config.allowed_origins,
    #         allow_credentials=True,
    #         allow_methods=["*"],
    #         allow_headers=["*"],
    #     )

    def setup_cors(self):
        """Configure CORS for the application."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )

        # Add these origins to your Config class
        self.config.allowed_origins = [
            "http://localhost:8000",
            "http://localhost",
            "http://127.0.0.1:8000",
            "http://127.0.0.1",
            "null",  # For Postman WebSocket requests
            "*"  # Allow all origins
        ]

    def setup_routes(self):
        """Set up WebSocket and HTTP routes."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)

    def configure_endpoints(self):
        """Configure additional HTTP endpoints."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "connections": len(self.connection_manager.connections),
                "version": "1.0.0"
            }

        @self.app.get("/stats")
        async def get_stats():
            """Get system statistics."""
            try:
                metrics = await self.metrics.get_metrics()
                return {
                    **metrics,
                    "active_connections": len(self.connection_manager.connections),
                    "uptime": str(datetime.now() - self._start_time)
                }
            except Exception as e:
                self.logger.error(f"Error getting stats: {str(e)}")
                raise HTTPException(status_code=500, detail="Error retrieving statistics")

        @self.app.post("/config")
        async def update_config(config: dict):
            """Update system configuration."""
            try:
                if 'logging_level' in config:
                    logging.getLogger().setLevel(config['logging_level'])

                if 'max_connections' in config:
                    self.config.max_connections = config['max_connections']

                return {"status": "configuration updated"}
            except Exception as e:
                self.logger.error(f"Error updating config: {str(e)}")
                raise HTTPException(status_code=500, detail="Error updating configuration")

    async def handle_websocket_connection(self, websocket: WebSocket):
        """Handle individual WebSocket connections."""
        connection_id = None
        try:
            await websocket.accept()
            connection_id = await self.connection_manager.add_connection(websocket)
            await self.metrics.increment_connections()

            while True:
                message = await websocket.receive_json()
                await self.metrics.increment_messages()
                await self.handle_client_message(websocket, message)

        except WebSocketDisconnect:
            if connection_id:
                await self.connection_manager.remove_connection(connection_id)
        except Exception as e:
            self.logger.error(f"Error handling WebSocket connection: {str(e)}")
            if connection_id:
                await self.connection_manager.remove_connection(connection_id)

    async def handle_client_message(self, websocket: WebSocket, message: dict):
        """Handle messages received from the client."""
        command = message.get('command')
        params = message.get('params', {})

        try:
            if command == '/new':
                await self.handle_new_chat(websocket, params)
            elif command == '/switch':
                await self.handle_switch_chat(websocket, params)
            elif command == '/list':
                await self.handle_list_chats(websocket)
            elif command == '/message':
                await self.handle_chat_message(websocket, params)
            elif command == '/upload':
                await self.handle_file_upload(websocket, params)
            elif command == '/upload_folder':
                await self.handle_folder_upload(websocket, params)
            elif command == '/files':
                await self.handle_list_files(websocket)
            else:
                await self.send_error(websocket, f"Unknown command: {command}")

        except Exception as e:
            self.logger.error(f"Error handling command {command}: {str(e)}")
            await self.send_error(websocket, str(e))

    async def handle_new_chat(self, websocket: WebSocket, params: dict):
        """Handle creation of new chat."""
        title = params.get('title')
        chat = await self.chat_manager.create_chat(title)
        await websocket.send_json({
            'type': 'chat_created',
            'chat_id': chat.chat_id,
            'title': chat.title
        })

    async def handle_switch_chat(self, websocket: WebSocket, params: dict):
        """Handle switching between chats."""
        chat_id = params.get('chat_id')
        if not chat_id:
            await self.send_error(websocket, "Chat ID required")
            return

        chat = await self.chat_manager.get_chat(chat_id)
        if not chat:
            await self.send_error(websocket, f"Chat not found: {chat_id}")
            return

        messages = await chat.get_chat_history()
        await websocket.send_json({
            'type': 'chat_history',
            'chat_id': chat_id,
            'messages': [
                {
                    'content': msg['content'],
                    'role': msg['role'],
                    'timestamp': msg.get('timestamp', datetime.now().isoformat())
                }
                for msg in messages
            ]
        })

    async def handle_chat_message(self, websocket: WebSocket, params: dict):
        """Handle new chat messages."""
        chat_id = params.get('chat_id')
        content = params.get('content')

        if not chat_id or not content:
            await self.send_error(websocket, "Chat ID and content required")
            return

        chat = await self.chat_manager.get_chat(chat_id)
        if not chat:
            await self.send_error(websocket, f"Chat not found: {chat_id}")
            return

        try:
            # Send user message
            await websocket.send_json({
                'type': 'chat_message',
                'message': {
                    'content': content,
                    'role': 'user',
                    'timestamp': datetime.now().isoformat()
                }
            })

            # Generate and send response
            response = await chat.generate_response(content)
            formatted_response = self.message_formatter.format_message({
                'content': response[0],
                'role': 'assistant'
            })

            await websocket.send_json({
                'type': 'chat_message',
                'message': {
                    'content': formatted_response.formatted_content,
                    'role': 'assistant',
                    'timestamp': datetime.now().isoformat()
                }
            })
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            await self.send_error(websocket, "Error generating response")

    async def handle_list_chats(self, websocket: WebSocket):
        """Handle listing all chats."""
        chats = await self.chat_manager.list_chats()
        await websocket.send_json({
            'type': 'chat_list',
            'chats': [
                {
                    'chat_id': chat.chat_id,
                    'title': chat.title,
                    'last_updated': chat.last_updated.isoformat()
                }
                for chat in chats
            ]
        })

    async def handle_file_upload(self, websocket: WebSocket, params: dict):
        """Handle file upload and indexing."""
        file_path = '/Users/ojasvsingh/personal_projects/rag_based_chatbot/src/crustdata_docs/'+params.get('file_name')
        if not file_path:
            await self.send_error(websocket, "File path required")
            return

        try:
            await websocket.send_json({
                'type': 'indexing_status',
                'status': 'started',
                'file_path': file_path
            })

            file_id = await self.index_manager.index_file(file_path)

            if file_id:
                await websocket.send_json({
                    'type': 'indexing_status',
                    'status': 'completed',
                    'file_path': file_path,
                    'file_id': file_id
                })
                await self.notify_file_update(file_id)
            else:
                await self.send_error(websocket, f"Failed to index file: {file_path}")

        except Exception as e:
            self.logger.error(f"Error indexing file: {str(e)}")
            await self.send_error(websocket, f"Error indexing file: {str(e)}")

    async def handle_folder_upload(self, websocket: WebSocket, params: dict):
        """Handle folder upload and indexing."""
        folder_path = params.get('folder_path')
        if not folder_path:
            await self.send_error(websocket, "Folder path required")
            return

        try:
            await websocket.send_json({
                'type': 'indexing_status',
                'status': 'started',
                'folder_path': folder_path
            })

            # for each file in the folder call handle_file_upload
            for file in os.listdir(folder_path):
                if file.endswith(".md"):
                    await self.handle_file_upload(websocket, {'file_name': file})

        except Exception as e:
            self.logger.error(f"Error indexing folder: {str(e)}")
            await self.send_error(websocket, f"Error indexing folder: {str(e)}")

    async def handle_list_files(self, websocket: WebSocket):
        """Handle listing indexed files."""
        try:
            files = await self.index_manager.get_indexed_files_stats()
            await websocket.send_json({
                'type': 'file_list',
                'files': [
                    {
                        'file_id': file['file_id'],
                        'file_path': str(file['file_path']),
                        'file_type': file['file_type'],
                        'embedding_model': str(file['embedding_model']),
                        'indexed_at': file['indexed_at'].isoformat()
                    }
                    for file in files
                ]
            })
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            await self.send_error(websocket, "Error retrieving file list")

    async def send_error(self, websocket: WebSocket, message: str):
        """Send error message to client."""
        try:
            await websocket.send_json({
                'type': 'error',
                'message': message
            })
        except Exception as e:
            self.logger.error(f"Error sending error message: {str(e)}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        try:
            await self.connection_manager.broadcast(message)
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {str(e)}")
            await self.metrics.increment_failed_messages()

    async def notify_chat_update(self, chat_id: str):
        """Notify all clients about a chat update."""
        try:
            chat = await self.chat_manager.get_chat(chat_id)
            if chat:
                await self.broadcast({
                    'type': 'chat_updated',
                    'chat': {
                        'chat_id': chat.chat_id,
                        'title': chat.title,
                        'last_updated': chat.last_updated.isoformat()
                    }
                })
        except Exception as e:
            self.logger.error(f"Error notifying chat update: {str(e)}")

    async def notify_file_update(self, file_id: str):
        """Notify all clients about a file update."""
        try:
            files = await self.index_manager.get_indexed_files_stats()
            file_info = next((f for f in files if f['file_id'] == file_id), None)
            if file_info:
                await self.broadcast({
                    'type': 'file_updated',
                    'file': {
                        'file_id': file_info['file_id'],
                        'file_path': str(file_info['file_path']),
                        'file_type': file_info['file_type'],
                        'embedding_model': str(file_info['embedding_model']),
                        'indexed_at': file_info['indexed_at'].isoformat()
                    }
                })
        except Exception as e:
            self.logger.error(f"Error notifying file update: {str(e)}")

    async def cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.config.ping_interval)
                stale_connections = [
                    conn_id for conn_id, conn in self.connection_manager.connections.items()
                    if conn.is_stale(self.config.connection_timeout)
                ]
                for conn_id in stale_connections:
                    await self.connection_manager.remove_connection(conn_id)
            except Exception as e:
                self.logger.error(f"Error cleaning up stale connections: {str(e)}")

    async def start(self):
        """Start the WebSocket server and background tasks."""
        cleanup_task = asyncio.create_task(self.cleanup_stale_connections())

        try:
            import uvicorn
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    async def stop(self):
        """Stop the WebSocket server and clean up."""
        for connection in self.connection_manager.connections.values():
            try:
                await connection.websocket.close()
            except Exception:
                pass
        self.connection_manager.connections.clear()

    def get_application(self):
        """Get the FastAPI application instance."""
        return self.app


# Fix 12: Update WebUIMetrics with async methods
class WebUIMetrics:
    """Track metrics for the Web UI."""

    def __init__(self):
        self.total_messages = 0
        self.total_connections = 0
        self.failed_messages = 0
        self.active_chats = 0
        self._start_time = datetime.now()
        self._lock = asyncio.Lock()

    async def increment_messages(self):
        async with self._lock:
            self.total_messages += 1

    async def increment_connections(self):
        async with self._lock:
            self.total_connections += 1

    async def increment_failed_messages(self):
        async with self._lock:
            self.failed_messages += 1

    async def get_metrics(self) -> dict:
        """Get current metrics."""
        async with self._lock:
            return {
                "total_messages": self.total_messages,
                "total_connections": self.total_connections,
                "failed_messages": self.failed_messages,
                "active_chats": self.active_chats,
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
            }


# if __name__ == "__main__":
#     # Example usage
#     import asyncio
#     from chat import ChatManager
#     from indexing.index_manager import IndexManager
#     from utils.message_formatter import MessageFormatter
#
#
#     async def main():
#         logging.basicConfig(level=logging.INFO)
#
#         chat_manager = ChatManager()
#         index_manager = IndexManager()
#         message_formatter = MessageFormatter()
#
#         bridge = WebUIBridge(
#             chat_manager=chat_manager,
#             index_manager=index_manager,
#             message_formatter=message_formatter,
#             host="localhost",
#             port=8000
#         )
#
#         try:
#             await bridge.start()
#         except KeyboardInterrupt:
#             await bridge.stop()
#         except Exception as e:
#             logging.error(f"Error running server: {str(e)}")
#
#
#     asyncio.run(main())