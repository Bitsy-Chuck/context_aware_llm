from typing import Dict, Optional
from rich.text import Text
from rich.style import Style
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime


class UIFormatter:
    """Helper class for consistent UI formatting."""

    def __init__(self):
        self.role_styles = {
            "user": Style(color="blue", bold=True),
            "assistant": Style(color="green", bold=True),
            "system": Style(color="yellow", bold=True)
        }

    def format_chat_header(self, chat_id: str, title: Optional[str] = None) -> Panel:
        """Format chat header panel."""
        header_text = Text()
        header_text.append("Chat: ", style="bold")
        header_text.append(chat_id[:8], style="blue")
        if title:
            header_text.append(f" - {title}", style="italic")

        return Panel(header_text, style="bold")

    def format_file_info(self, file_info: Dict) -> Table:
        """Format file information as a table."""
        table = Table(show_header=False, box=None)
        table.add_column("Property")
        table.add_column("Value")

        table.add_row("Name", file_info["name"])
        table.add_row("Type", file_info["mime_type"])
        table.add_row("Size", self._format_size(file_info["size"]))
        table.add_row("Created", file_info["created_at"].strftime("%Y-%m-%d %H:%M:%S"))

        return table

    def _format_size(self, size_in_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.1f} {unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.1f} TB"