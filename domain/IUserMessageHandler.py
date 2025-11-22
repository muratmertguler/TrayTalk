from abc import ABC, abstractmethod
from datetime import datetime


class IUserMessageHandler(ABC):
    @abstractmethod
    async def message_handler(self, user_id: str, session_id: str, message: dict, timestamp: datetime) -> dict:
        pass
        """
         Process a user message from JSON data and return a response.
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            message_data: A dictionary containing the message data, including the content as a string to be parsed
            timestamp: The time the message was sent, as a datetime timestamp
            return: A tuple containing a success flag and an optional response message
        """
        

