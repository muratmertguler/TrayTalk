from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from langchain_community.chat_message_histories import ChatMessageHistory
from domain.models import Conversation

class ILLMClient(ABC):
    @abstractmethod
    def response_generation(self, query: str, session_id: str = "default") -> str:
        pass

class IRerankerClient(ABC):
    @abstractmethod
    def score(self, query: str, doc: str, instruction: Optional[str] = None) -> float:
        pass

    @abstractmethod
    def score_batch(self, pairs: List[Tuple[str, str]], instruction: Optional[str] = None, batch_size: int = 8) -> List[float]:
        pass

class IMemoryManager(ABC):
    @abstractmethod
    def add_messages_to_history(self, messages: List[Dict[str, str]], session_id: str) -> None:
        pass

    @abstractmethod
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        pass

    @abstractmethod
    def clear_history(self, session_id: str = "default"):
        pass


class IConversationRepository(ABC):
    @abstractmethod
    async def save_conversation(self, conversation:Conversation):
        pass

    @abstractmethod
    async def get_conversation(self, session_id: str) -> Conversation:
        pass