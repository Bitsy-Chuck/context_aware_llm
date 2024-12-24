import concurrent
import os
import traceback
from datetime import datetime
from pathlib import Path
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
            await self.list_chats()
            choice = await self.prompt_session.prompt_async(
                "Enter chat ID to switch or press Enter for new chat: "
            )

            if choice.strip():
                await self.switch_chat(choice)
            else:
                await self.create_new_chat()
            # choose any chat, if name is not found, create a new one with that name

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
        self.logger.info("--> ", chats)
        table = Table(title="Available Chats")
        table.add_column("Chat ID")
        table.add_column("Title")
        table.add_column("Last Updated")

        for chat in chats:
            table.add_row(
                chat.chat_id[:8],
                chat.title,
                chat.last_updated.strftime("%Y-%m-%d %H:%M")
            )

        self.console.print(table)

    import os
    from pathlib import Path

    import concurrent.futures
    from datetime import datetime

    async def upload_file(self):
        """Upload and index a file or all files in a directory using multiple threads."""
        file_path = await self.prompt_session.prompt_async("Enter file/folder path: ")
        path = Path(file_path)
        failed_files = []

        try:
            file_ids = []
            with Progress() as progress:
                task = progress.add_task("Indexing...", total=None)

                if path.is_dir():
                    self.console.print(f"[green]Starting indexing of directory: {file_path}[/green]")

                    # Collect all files first
                    files_to_process = []
                    for root, _, files in os.walk(path):
                        for file in files:
                            full_path = Path(root) / file
                            files_to_process.append(str(full_path))

                    async def process_single_file(file_path):
                        try:
                            self.console.print(f"[blue]Indexing file: {file_path}[/blue]")
                            file_info = self.file_helper.get_file_info(file_path)
                            if not file_info:
                                file_id = await self.index_manager.index_file(file_path)
                                if file_id:
                                    return {"success": True, "file_id": file_id, "file_path": file_path}
                                return {"success": False, "error": "No file ID returned", "file_path": file_path}
                        except Exception as file_error:
                            self.logger.error(f"Error indexing file {file_path}: {str(file_error)}")
                            self.console.print(f"[yellow]Skipping file {file_path}: {str(file_error)}[/yellow]")
                            return {
                                "success": False,
                                "error": str(file_error),
                                "file_path": file_path
                            }

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        tasks = [
                            asyncio.create_task(process_single_file(file_path))
                            for file_path in files_to_process
                        ]

                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Process results and collect failed files
                        for result in results:
                            if isinstance(result, Exception):
                                failed_files.append({
                                    "file_path": "Unknown",
                                    "error": str(result)
                                })
                            elif result["success"]:
                                file_ids.append(result["file_id"])
                            else:
                                failed_files.append({
                                    "file_path": result["file_path"],
                                    "error": result["error"]
                                })

                    progress.update(task, completed=True)
                    self.console.print(
                        f"[green]Successfully indexed {len(file_ids)} files from directory: {file_path}[/green]")

                else:
                    # Handle single file
                    try:
                        self.console.print(f"[green]Starting indexing of file: {file_path}[/green]")
                        file_id = await self.index_manager.index_file(str(path))
                        if file_id:
                            file_ids.append(file_id)
                        else:
                            failed_files.append({
                                "file_path": str(path),
                                "error": "No file ID returned"
                            })
                    except Exception as e:
                        failed_files.append({
                            "file_path": str(path),
                            "error": str(e)
                        })
                    progress.update(task, completed=True)
                    self.console.print(f"[green]Successfully indexed file: {file_path}[/green]")

                # Save failed files to a txt file if any failures occurred
                if failed_files:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    failed_files_path = f"failed_indexing_{timestamp}.txt"

                    with open(failed_files_path, "w") as f:
                        f.write("Failed Files Report\n")
                        f.write("=================\n")
                        f.write(f"Attempted indexing at: {datetime.now()}\n")
                        f.write(f"Source directory/file: {file_path}\n\n")

                        for failed in failed_files:
                            f.write(f"File: {failed['file_path']}\n")
                            f.write(f"Error: {failed['error']}\n")
                            f.write("-" * 50 + "\n")

                    self.console.print(
                        f"[yellow]{len(failed_files)} files failed to index. "
                        f"Details saved to: {failed_files_path}[/yellow]"
                    )

                return file_ids

        except Exception as e:
            self.logger.error(f"Error processing path {file_path}: {str(e)}")
            self.console.print(f"[red]Error processing path: {str(e)}[/red]")
            return []

    async def list_files(self):
        """List all indexed files."""
        files = await self.index_manager.get_indexed_files_stats()

        table = Table(title="Indexed Files")
        table.add_column("File ID")
        table.add_column("Path")
        table.add_column("Type")
        # table.add_column("Chunks")
        table.add_column("Embedding Model")
        table.add_column("Indexed At")
        #
        # try:
        #     self.logger.info("files", files)
        #     self.logger.info("files1", files[0]['file_id'][:8])
        #     self.logger.info("files2", str(files[0]['file_path']))
        #     self.logger.info("files3", files[0]['embedding_model'])
        #     # self.logger.info("files4", str(files[0]['metadata']['chunk_']))
        #     self.logger.info("files5", files[0]['indexed_at'].strftime("%Y-%m-%d %H:%M"))
        # except Exception as e:
        #     self.logger.error("Error: ", traceback.print_stack())

        for file in files:
            self.logger.info("files6", file)

            table.add_row(
                file['file_id'][:8],
                str(file['file_path']),
                file['file_type'],
                str(file['embedding_model']),
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