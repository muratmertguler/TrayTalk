from domain.interfaces import IMemoryManager
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict

class MemoryManager(IMemoryManager):
    def __init__(self):
        self.store: dict[str, ChatMessageHistory] = {} 

    def add_messages_to_history(self, messages: List[Dict[str, str]], session_id: str) -> None:
        history = self.get_session_history(session_id)
        
        for msg in messages:
            if msg["role"] == "system":
                history.add_message(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                history.add_message(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.add_message(AIMessage(content=msg["content"]))

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_recent_messages(self, session_id: str, max_context: int) -> List:
        history = self.get_session_history(session_id)
        return history.messages[-max_context:]

    def clear_history(self, session_id: str = "default"):
        if session_id in self.store:
            self.store[session_id].clear()