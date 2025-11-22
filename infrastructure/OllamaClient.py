from domain.ILLMClient import ILLMClient

from typing import Optional, AsyncIterator
import asyncio

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_ollama import ChatOllama


class OllamaClient(ILLMClient):
    def __init__(self, model_name: str, url: str, temperature: float = 0.2, 
         max_input_tokens: int = 512, max_output_tokens: int = 512,
         api_key: Optional[str] = None):

        self.url = url
        self.api_key           = api_key
        self.model_name        = model_name
        self.temperature       = temperature
        self.max_input_tokens  = max_input_tokens
        self.max_output_tokens = max_output_tokens
        
        self.ollama_client = ChatOllama(
            model = self.model_name,
            base_url = self.url,
            temperature = self.temperature,
            max_input_tokens = self.max_input_tokens,
            max_response_tokens = self.max_output_tokens
        )
        
    async def generate_text(self, prompt: str) -> str:
        response = self.ollama_client.invoke(prompt)
        return response
    
    async def generate_text_stream(self, query: str, system_prompt: Optional[str] = None, 
                                   rag_information: Optional[str] = None) -> AsyncIterator[str]:
        """
        Async generator that yields chunks from Ollama streaming.
        Adapts sync ChatOllama.stream() to async context via executor.
        """
        messages = []
        if system_prompt:
            rag_docs = "system prompts: " + system_prompt
            if rag_information:
                rag_docs += "\nRetrieval Docs: " + rag_information
            messages.append(SystemMessage(content=rag_docs))
        messages.append(HumanMessage(content=query))
        
        # Run sync stream in thread pool to not block event loop
        loop = asyncio.get_running_loop()
        
        # Get the sync generator in executor
        def _get_stream_iter():
            return self.ollama_client.stream(messages)
        
        stream_iter = await loop.run_in_executor(None, _get_stream_iter)
        
    

    async def enrich_question(self, prompt: str, text: str) -> str:
        pass