from typing import Dict, List, Optional
import re
from dataclasses import dataclass
from datetime import datetime

from ..utils.markdown_helper import MarkdownHelper


@dataclass
class FormattedMessage:
    content: str
    timestamp: datetime
    role: str
    metadata: Optional[Dict] = None
    formatted_content: Optional[str] = None


class MessageFormatter:
    def __init__(self, markdown_helper: MarkdownHelper = None):
        self.markdown_helper = markdown_helper or MarkdownHelper()
        self.role_colors = {
            "user": "\033[94m",  # Blue
            "assistant": "\033[92m",  # Green
            "system": "\033[93m"  # Yellow
        }
        self.reset_color = "\033[0m"

    def format_message(
            self,
            message: Dict,
            terminal_width: int = 80,
            show_metadata: bool = False
    ) -> FormattedMessage:
        """Format a message for display."""
        # Basic message info
        formatted = FormattedMessage(
            content=message["content"],
            timestamp=message.get("timestamp", datetime.now()),
            role=message["role"],
            metadata=message.get("metadata")
        )

        # Format content
        content = message["content"]
        content = self.markdown_helper.sanitize_markdown(content)
        content = self.markdown_helper.format_response(content, terminal_width)

        # Apply syntax highlighting to code blocks
        code_blocks = self.markdown_helper.extract_code_blocks(content)
        for block in code_blocks:
            highlighted_code = self.markdown_helper.highlight_code(block)
            content = content.replace(
                f"```{block.language}\n{block.code}\n```",
                highlighted_code
            )

        # Add role color
        role_color = self.role_colors.get(message["role"], "")
        header = f"{role_color}{message['role'].capitalize()}{self.reset_color}"
        timestamp = formatted.timestamp.strftime("%H:%M:%S")

        # Format final content
        formatted_parts = [
            f"{header} [{timestamp}]:",
            content
        ]

        # Add metadata if requested
        if show_metadata and formatted.metadata:
            meta_lines = []
            if formatted.metadata.get("images"):
                meta_lines.append(f"Images: {len(formatted.metadata['images'])}")
            if formatted.metadata.get("documents"):
                meta_lines.append(f"Documents: {len(formatted.metadata['documents'])}")
            if meta_lines:
                formatted_parts.append(
                    f"\n{role_color}Attachments:{self.reset_color} " +
                    ", ".join(meta_lines)
                )

        formatted.formatted_content = "\n".join(formatted_parts)
        return formatted

    def format_chat_preview(self, chat_info: Dict, max_length: int = 50) -> str:
        """Format chat preview for list display."""
        title = chat_info["title"]
        last_update = chat_info["last_updated"].strftime("%Y-%m-%d %H:%M")

        if len(title) > max_length:
            title = title[:max_length - 3] + "..."

        return f"{title} (Last updated: {last_update})"