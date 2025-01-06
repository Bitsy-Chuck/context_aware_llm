from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from .base_llm import BaseLLM, Message
import base64
import logging
import os
from pathlib import Path
import mimetypes
import traceback

from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
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
            model_name: str = "gpt-4",
            model_type: str = "azure_openai",
            api_key: Optional[str] = None,
            api_version: Optional[str] = None,
            api_base: Optional[str] = None,
            deployment_name: Optional[str] = None,
            audience: Optional[str] = None,
            organization: Optional[str] = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ):
        """
        Initialize the ChatGPT model.

        Args:
            model_name: Name of the GPT model to use
            type: Type of API to use ('azure_openai' or 'openai')
            api_key: API key
            api_version: API version (required for Azure)
            api_base: API base URL (required for Azure)
            deployment_name: Model deployment name (required for Azure)
            audience: Azure audience (optional)
            organization: Organization ID (optional)
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter (0-1)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.deployment_name = deployment_name

        # Initialize the client based on type
        if self.model_type == 'azure_openai':
            if not api_base or not api_version:
                raise ValueError("api_base and api_version are required for Azure OpenAI")

            self.client = AsyncAzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment="gpt-4o"
            )
        else:
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

    async def generate_response(
            self,
            messages: List[Message],
            context: Optional[str] = None,
            **kwargs
    ) -> str:
        """Generate a response using the ChatGPT model."""
        try:
            formatted_messages = []
            self.logger.info(f"Generating response with {len(messages)} messages")

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

            # Prepare model parameter based on type
            model_param = self.deployment_name if self.model_type == 'azure_openai' else self.model_name

            # Merge kwargs with default parameters
            request_params = {
                "model": model_param,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **kwargs
            }

            # Generate response
            response = await self.client.chat.completions.create(**request_params)

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