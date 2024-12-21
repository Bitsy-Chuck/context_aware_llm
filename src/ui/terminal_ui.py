import traceback
from typing import Optional, List, Dict
import asyncio
import logging
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress

from ..chat import ChatManager, ChatSession, MessageFormatter
from ..utils.file_helper import FileHelper
from ..indexing.index_manager import IndexManager


class TerminalUI:
    def __init__(
            self,
            chat_manager: ChatManager,
            index_manager: IndexManager,
            message_formatter: MessageFormatter
    ):
        self.chat_manager = chat_manager
        self.index_manager = index_manager
        self.message_formatter = message_formatter
        self.console = Console()
        self.current_chat: Optional[ChatSession] = None
        self.file_helper = FileHelper()
        self.logger = logging.getLogger(__name__)

        # Initialize prompt session
        self.prompt_style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'chat-id': '#0000aa',
            'command': '#aa0000',
        })

        self.prompt_session = PromptSession(
            style=self.prompt_style,
            complete_while_typing=True
        )

        # Available commands
        self.commands = {
            '/new': self.create_new_chat,
            '/switch': self.switch_chat,
            '/list': self.list_chats,
            '/clear': self.clear_screen,
            '/help': self.show_help,
            '/exit': self.exit_app,
            '/upload': self.upload_file,
            '/files': self.list_files,
            '/delete': self.delete_chat
        }

    async def start(self):
        """Start the terminal UI."""
        try:
            await self._show_welcome_message()
            await self.create_new_chat()
            await self._main_loop()
        except Exception as e:
            self.logger.error(f"Error in UI: {str(e)}")
            raise

    async def _show_welcome_message(self):
        """Display welcome message."""
        welcome_text = """
        # RAG-powered Chat System

        Type your message or use commands:
        - /new - Create new chat
        - /switch <chat_id> - Switch to another chat
        - /list - List all chats
        - /upload - Upload and index a file
        - /files - List indexed files
        - /help - Show help
        - /exit - Exit application
        """
        self.console.print(Markdown(welcome_text))

    async def _main_loop(self):
        """Main interaction loop."""
        while True:
            try:
                # Show chat context
                chat_prefix = f"[chat-id]{self.current_chat.chat_id[:8]}[/chat-id]" if self.current_chat else ""
                prompt_text = HTML(f"{chat_prefix}> ")

                # Get user input
                user_input = await self.prompt_session.prompt_async(prompt_text)

                if not user_input.strip():
                    continue

                if user_input.startswith('/'):
                    # Handle commands
                    command = user_input.split()[0]
                    args = user_input.split()[1:] if len(user_input.split()) > 1 else []

                    if command in self.commands:
                        await self.commands[command](*args)
                    else:
                        self.console.print("[red]Unknown command. Type /help for available commands.[/red]")
                else:
                    # Handle chat message
                    await self._handle_chat_message(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                self.console.print(f"[red]Error: {str(e)}[/red]")

    async def _handle_chat_message(self, message: str):
        """Handle user chat message."""
        if not self.current_chat:
            self.console.print("[red]No active chat. Create one with /new[/red]")
            return

        try:
            with Progress() as progress:
                task = progress.add_task("Generating response...", total=None)

                # Generate and display response
                response = await self.current_chat.generate_response(message)

                progress.update(task, completed=True)
                self.logger.info("------", response)
                # Format and display the response
                formatted_response = self.message_formatter.format_message({
                    "content": response[0],
                    "role": "assistant"
                })

                self.console.print(Markdown(formatted_response.formatted_content))

        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.logger.error("Error: ", traceback.print_stack())
            self.console.print(f"[red]Error generating response: {str(e)}[/red]")

    async def create_new_chat(self):
        """Create a new chat session."""
        title = await self.prompt_session.prompt_async("Enter chat title (optional): ")
        self.current_chat = await self.chat_manager.create_chat(title if title else None)
        self.console.print(f"[green]Created new chat: {self.current_chat.chat_id}[/green]")

    async def switch_chat(self, chat_id: str):
        """Switch to another chat session."""
        chat = await self.chat_manager.get_chat(chat_id)
        if chat:
            self.current_chat = chat
            self.console.print(f"[green]Switched to chat: {chat_id}[/green]")

            # Show recent messages
            messages = await chat.get_chat_history(limit=5)
            for msg in messages:
                formatted = self.message_formatter.format_message(msg)
                self.console.print(Markdown(formatted.formatted_content))
        else:
            self.console.print(f"[red]Chat not found: {chat_id}[/red]")

    async def list_chats(self):
        """List all chat sessions."""
        chats = await self.chat_manager.list_chats()

        table = Table(title="Available Chats")
        table.add_column("Chat ID")
        table.add_column("Title")
        table.add_column("Last Updated")

        for chat in chats:
            table.add_row(
                chat['chat_id'][:8],
                chat['title'],
                chat['last_updated'].strftime("%Y-%m-%d %H:%M")
            )

        self.console.print(table)

    async def upload_file(self):
        """Upload and index a file."""
        file_path = await self.prompt_session.prompt_async("Enter file path: ")

        try:
            with Progress() as progress:
                task = progress.add_task("Indexing file...", total=None)

                self.console.print(f"[green]Successfully started indexed file: {file_path}[/green]")
                # Index the file
                file_id = await self.index_manager.index_file(file_path)

                progress.update(task, completed=True)

                self.console.print(f"[green]Successfully indexed file: {file_path}[/green]")
                return file_id

        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            self.console.print(f"[red]Error uploading file: {str(e)}[/red]")

    async def list_files(self):
        """List all indexed files."""
        files = await self.index_manager.get_indexed_files_stats()

        table = Table(title="Indexed Files")
        table.add_column("File ID")
        table.add_column("Path")
        table.add_column("Type")
        table.add_column("Chunks")
        table.add_column("Indexed At")

        try:
            self.logger.info("files", files)
            self.logger.info("files1", files[0]['file_id'][:8])
            self.logger.info("files2", str(files[0]['file_path']))
            self.logger.info("files3", files[0]['file_type'])
            self.logger.info("files4", str(files[0]['metadata']['num_chunks']))
            self.logger.info("files5", files[0]['indexed_at'].strftime("%Y-%m-%d %H:%M"))
        except Exception as e:
            self.logger.error("Error: ", traceback.print_stack())

        for file in files:
            self.logger.info("files6", file)

            table.add_row(
                file['file_id'][:8],
                str(file['file_path']),
                file['file_type'],
                str(file['metadata']['num_chunks']),
                file['indexed_at'].strftime("%Y-%m-%d %H:%M")
            )

        self.console.print(table)

    async def delete_chat(self, chat_id: str):
        """Delete a chat session."""
        if await self.chat_manager.delete_chat(chat_id):
            if self.current_chat and self.current_chat.chat_id == chat_id:
                self.current_chat = None
            self.console.print(f"[green]Deleted chat: {chat_id}[/green]")
        else:
            self.console.print(f"[red]Error deleting chat: {chat_id}[/red]")

    def clear_screen(self):
        """Clear the terminal screen."""
        self.console.clear()

    def show_help(self):
        """Show help information."""
        help_text = """
        # Available Commands

        ## Chat Management
        - `/new` - Create a new chat session
        - `/switch <chat_id>` - Switch to another chat
        - `/list` - List all available chats
        - `/delete <chat_id>` - Delete a chat session

        ## File Management
        - `/upload` - Upload and index a new file
        - `/files` - List all indexed files

        ## Interface
        - `/clear` - Clear the screen
        - `/help` - Show this help message
        - `/exit` - Exit the application

        ## Usage
        - Just type your message to chat
        - Use code blocks with \```language\n code \```
        - Drag and drop files to upload them
        """
        self.console.print(Markdown(help_text))

    async def exit_app(self):
        """Exit the application."""
        self.console.print("[yellow]Goodbye![/yellow]")
        raise EOFError()