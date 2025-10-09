from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain.memory import ChatMessageHistory
from domain.models import Conversation

class ILLMClient(ABC):
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.memory_store = {}
        self.system_prompt = system_prompt

    @abstractmethod
    def response_generation(self, query: str, session_id: str = "default") -> str:
        pass

    @abstractmethod
    def _response_generation_without_memory(self, query:str) -> str:
        pass


class IMemoryManager(ABC):
    @abstractmethod
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        pass

    @abstractmethod
    def clear_history(self, session_id: str = "default"):
        pass

    @abstractmethod
    def get_conversation_history(self, session_id: str = "default") -> list[Dict[str, Any]]:
        pass


class IConversationRepository(ABC):
    @abstractmethod
    async def save_conversation(self, conversation:Conversation):
        pass

    @abstractmethod
    async def get_conversation(self, session_id: str) -> Conversation:
        pass
