from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from .base_llm import BaseLLM, Message
import base64
import logging
import os
from pathlib import Path
import mimetypes
import traceback


class ChatGPTModel(BaseLLM):
    """Implementation of BaseLLM for OpenAI's GPT models."""

    def __init__(
            self,
            model_name: str = "gpt-4-turbo-preview",
            api_key: Optional[str] = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            top_p: float = 0.9,
            organization: Optional[str] = None
    ):
        """
        Initialize the ChatGPT model.

        Args:
            model_name: Name of the GPT model to use
            api_key: OpenAI API key (optional, can use environment variable)
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter (0-1)
            organization: OpenAI organization ID (optional)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize the client
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization or os.getenv("OPENAI_ORG_ID")
        )

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Validate model name
        if not self._is_valid_model():
            raise ValueError(f"Invalid model name: {model_name}")

    def _is_valid_model(self) -> bool:
        """Check if the model name is valid."""
        valid_models = [
            "gpt-4-turbo-preview",
            "gpt-4-vision-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
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
        Process an image file into the format required by GPT-4 Vision.

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
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self._get_mime_type(str(img_path))};base64,{base64_image}"
                    }
                }
        except Exception as e:
            self.logger.error(f"Error processing image {img_path}: {str(e)}\n{traceback.format_exc()}")
            raise

    async def generate_response(
            self,
            messages: List[Message],
            context: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        Generate a response using the ChatGPT model.

        Args:
            messages: List of Message objects containing the conversation history
            context: Optional context string (e.g., for RAG applications)
            **kwargs: Additional keyword arguments to pass to the API

        Returns:
            str: The generated response text
        """
        try:
            formatted_messages = []
            self.logger.info(f"Generating response with {len(messages)} messages", messages)
            # Add RAG context if provided
            if context:
                formatted_messages.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": f"Use the following context to answer questions:\n{context}"
                    }]
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

                # Handle images if present and model supports vision
                if msg.images and self.supports_vision():
                    for img_path in msg.images:
                        try:
                            image_part = await self._process_image(img_path)
                            content_parts.append(image_part)
                        except Exception as e:
                            self.logger.error(f"Error processing image {img_path}: {str(e)}")
                            continue

                # Build the message content
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    message_content = content_parts[0]["text"]
                else:
                    message_content = content_parts

                # Add the formatted message
                formatted_messages.append({
                    "role": msg.role,
                    "content": message_content
                })

            self.logger.info(f"Sending request with {len(formatted_messages)} messages")

            # Merge kwargs with default parameters
            request_params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **kwargs
            }
            self.logger.info(request_params)

            # Generate response
            response = await self.client.chat.completions.create(**request_params)

            self.logger.info("---asdasd----", response)
            # Extract and return the text content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("Empty response received from ChatGPT")

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RuntimeError(error_msg)

    def supports_vision(self) -> bool:
        """Check if the model supports vision/image input."""
        return self.model_name in ["gpt-4-vision-preview"]

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
        return "You are ChatGPT, an AI assistant created by OpenAI. " \
               "You are direct, helpful, and honest."

    def _get_max_context_length(self) -> int:
        """Get the maximum context length for the current model."""
        context_lengths = {
            "gpt-4-turbo-preview": 128000,
            "gpt-4-vision-preview": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385
        }
        return context_lengths.get(self.model_name, 4096)