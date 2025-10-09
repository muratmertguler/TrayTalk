from domain.interfaces import IMemoryManager
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any

class MemoryManager(IMemoryManager):
    def __init__(self):
        self.store: dict[str, ChatMessageHistory] = {}

    def clear_history(self, session_id: str = "default"):
        if session_id in self.store:
            self.store[session_id].clear()

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        if session_id not in self.store:
            return []
        messages = []
        for msg in self.store[session_id].messages:
            if isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages