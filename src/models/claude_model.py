from typing import List, Optional, Dict, Any
import anthropic
from .base_llm import BaseLLM, Message
import base64
import logging
import os
from pathlib import Path
import mimetypes


class ClaudeModel(BaseLLM):
    """Implementation of BaseLLM for Anthropic's Claude models."""

    def __init__(
            self,
            model_name: str = "claude-3-sonnet-20240229",
            api_key: Optional[str] = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            top_p: float = 0.9
    ):
        """
        Initialize the Claude model.

        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key (optional, can use environment variable)
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter (0-1)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize the client
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Validate model name
        if not self._is_valid_model():
            raise ValueError(f"Invalid model name: {model_name}")

    def _is_valid_model(self) -> bool:
        """Check if the model name is valid."""
        valid_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]
        return self.model_name in valid_models

    def _get_mime_type(self, file_path: str) -> str:
        """
        Get the MIME type of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: MIME type of the file
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Default to JPEG if can't determine
            return "image/jpeg"
        return mime_type

    async def _process_image(self, img_path: str) -> Dict[str, Any]:
        """
        Process an image file into the format required by Claude.

        Args:
            img_path: Path to the image file

        Returns:
            Dict containing the formatted image data
        """
        try:
            img_path = Path(img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

            with open(img_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode()

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self._get_mime_type(str(img_path)),
                        "data": base64_image
                    }
                }
        except Exception as e:
            self.logger.error(f"Error processing image {img_path}: {str(e)}")
            raise

    async def generate_response(
            self,
            messages: List[Message],
            context: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        Generate a response using the Claude model.

        Args:
            messages: List of Message objects containing the conversation history
            context: Optional context string (e.g., for RAG applications)
            **kwargs: Additional keyword arguments to pass to the API

        Returns:
            str: The generated response text
        """
        try:
            formatted_messages = []

            # Add RAG context if provided
            if context:
                formatted_messages.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Use the following context to answer questions:\n{context}"
                        }
                    ]
                })

            # Format each message
            for msg in messages:
                content_parts = []

                # Handle text content
                if msg.content:
                    content_parts.append({
                        "type": "text",
                        "text": msg.content
                    })

                # Handle images if present
                if msg.images:
                    for img_path in msg.images:
                        try:
                            image_part = await self._process_image(img_path)
                            content_parts.append(image_part)
                        except Exception as e:
                            self.logger.error(f"Error processing image {img_path}: {str(e)}")
                            continue

                # Only add the message if there are content parts
                if content_parts:
                    formatted_messages.append({
                        "role": msg.role,
                        "content": content_parts
                    })

            self.logger.debug(f"Sending request with {len(formatted_messages)} messages")

            # Merge kwargs with default parameters
            request_params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **kwargs
            }

            # Generate response
            response = await self.client.messages.create(**request_params)

            # Extract and return the text content
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise ValueError("Empty response received from Claude")

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def supports_vision(self) -> bool:
        """Check if the model supports vision/image input."""
        return "claude-3" in self.model_name.lower()

    def supports_documents(self) -> bool:
        """Check if the model supports document input."""
        return True

    def get_model_name(self) -> str:
        """Get the name of the current model."""
        return self.model_name

    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens supported."""
        return self.max_tokens

    async def get_system_prompt(self) -> str:
        """Get the default system prompt for the model."""
        return "You are Claude, an AI assistant created by Anthropic. " \
               "You are direct, helpful, and honest."