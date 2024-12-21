import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class CodeBlock:
    language: str
    code: str
    start_line: int
    end_line: int


class MarkdownHelper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self._inline_code_pattern = re.compile(r'`([^`]+)`')

    def extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """Extract all code blocks from markdown text."""
        blocks = []
        for match in self._code_block_pattern.finditer(text):
            language = match.group(1) or 'text'
            code = match.group(2)
            start_pos = match.start()
            end_pos = match.end()

            # Calculate line numbers
            start_line = text[:start_pos].count('\n') + 1
            end_line = start_line + code.count('\n') + 1

            blocks.append(CodeBlock(
                language=language,
                code=code,
                start_line=start_line,
                end_line=end_line
            ))
        return blocks

    def format_response(self, text: str, terminal_width: int = 80) -> str:
        """Format markdown text for terminal display."""
        lines = []
        current_line = ""

        # Split into lines preserving code blocks
        for line in text.split('\n'):
            # Check if this is a code block marker
            if line.startswith('```'):
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                lines.append(line)
                continue

            # Word wrap normal text
            words = line.split()
            for word in words:
                if len(current_line) + len(word) + 1 <= terminal_width:
                    current_line += (" " + word if current_line else word)
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)
                current_line = ""

        return '\n'.join(lines)

    def highlight_code(self, code_block: CodeBlock) -> str:
        """Apply syntax highlighting to code block."""
        try:
            import pygments
            from pygments.lexers import get_lexer_by_name
            from pygments.formatters import Terminal256Formatter

            lexer = get_lexer_by_name(code_block.language, stripall=True)
            formatter = Terminal256Formatter()

            return pygments.highlight(
                code_block.code,
                lexer,
                formatter
            )
        except Exception as e:
            self.logger.warning(f"Could not apply syntax highlighting: {str(e)}")
            return code_block.code

    def sanitize_markdown(self, text: str) -> str:
        """Clean and sanitize markdown text."""
        # Remove multiple consecutive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Ensure proper spacing around headers
        text = re.sub(r'(#{1,6})\s*(.+?)\s*(?:\n|$)', r'\n\1 \2\n', text)

        # Ensure proper list formatting
        text = re.sub(r'^\s*[-*+]\s+', '* ', text, flags=re.MULTILINE)

        return text.strip()