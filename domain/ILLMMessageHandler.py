from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime


class ILLMMessageHandler(ABC):
    @abstractmethod
    async def message_handler(self, role: str, content: str, metadata: Optional[dict], timestamp: datetime) -> dict:
        pass
        """ 
         Process a LLM message from JSON or streaming data and return a response.
            role: Role of the message sender (e.g., user, assistant)
            content: The content of the LLM message
            metadata: Additional metadata for the message
            timestamp: The time the message was sent, as a datetime timestamp
            return: A tuple containing a success flag and an optional response message
        """
