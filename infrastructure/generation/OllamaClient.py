from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from domain.interfaces import IMemoryManager
from infrastructure.memory import MemoryManager

from domain.interfaces import ILLMClient


class OllamaClient(ILLMClient):
    def __init__(self, memory_manager: IMemoryManager,
                 model: str, base_url: str,
                 temperature: float = 0.7, max_tokens: int = 1000):
        self.memory = memory_manager
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = ChatOllama(model=self.model,
                            base_url=self.base_url,
                            temperature=temperature,
                            max_tokens=max_tokens)
        

    async def response_generation(self, messages: List[Dict[str, str]],
                                session_id: str = "default",
                                max_context: int = 5,
                                **kwargs):
        
        self.memory.add_messages_to_history(messages, session_id)
        recent_messages = self.memory.get_recent_messages(session_id, max_context)
        langchain_messages = [(m.type, m.content) for m in recent_messages]
        prompt = ChatPromptTemplate.from_messages(langchain_messages)
        chain = prompt | self.llm
        
        async for chunk in chain.astream({}):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)
