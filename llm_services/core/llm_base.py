from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from datetime import datetime
from socket import gethostbyname


class LLMRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    session_id: str = Field("default", description="Session ID")
    user_id: Optional[str] = Field(None, description="User Hostname") 


class LLMResponse(BaseModel):
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="AI answer")
    session_id: str = Field(..., description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(True)
    model_used: Optional[str] = Field(None, description="LLM model name")


class LLMClient(ABC):
    def __init__(self, llm_name:str, url:str):
        self.url = url
        self.llm_name = llm_name
        self.llm = None
        self.memory_store: Dict[str, ChatMessageHistory] = {}
        self.system_prompt = "you are a helpful assistant"

    @abstractmethod
    def generate_response(self, query:str, session_id:str="default") -> str:
        pass

    def _generate_response_without_memory(self, query:str) -> str:
        messages = [
            SystemMessage(content="you are helpful assistant"),
            HumanMessage(content=query)
        ]
        response = self.llm.invoke(messages)
        if hasattr(response, "content"):
            return str(response.content)
        else:
            return str(response)

    def get_session_history(self, session_id:str) -> ChatMessageHistory:
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ChatMessageHistory()
        return self.memory_store[session_id]

    def clear_memory(self, session_id:str="default"):
        if session_id in self.memory_store:
            self.memory_store[session_id].clear()

    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        if session_id not in self.memory_store:
            return []
        
        messages = []
        for msg in self.memory_store[session_id].messages:
            if isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        return messages
    
    def set_system_prompt(self, system_prompt:str):
        self.system_prompt = system_prompt

    def get_llm_name(self) -> str:
        return self.llm_name