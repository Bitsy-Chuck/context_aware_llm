from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str
    images: Optional[List[str]] = None
    documents: Optional[List[str]] = None


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    @abstractmethod
    async def generate_response(
            self,
            messages: List[Message],
            context: Optional[str] = None,
            **kwargs
    ) -> str:
        """Generate a response given a list of messages and optional context."""
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether the model supports processing images."""
        pass

    @abstractmethod
    def supports_documents(self) -> bool:
        """Whether the model can directly process documents."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name/identifier of the model."""
        pass