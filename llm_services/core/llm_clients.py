import os
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_services.core.llm_base import LLMClient
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = None, model_name: str = None):
        base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        model_name = model_name or os.getenv("LLM_MODEL", "llama3.2:latest")
        super().__init__(llm_name=model_name, url=base_url)
        self.llm = ChatOllama(model=model_name,base_url=base_url)
        self.memory_store = {}
        self.system_prompt = "you are a assistance"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history")
        ])
        self.chain = self.prompt|self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain, self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def generate_response(self, query:str, session_id:str="default") -> str:
        if session_id in self.memory_store:
            try:
                response = self.chain_with_history.invoke(
                    {"input":query},
                    config={"configurable":{"session_id":session_id}}
                )
                return response.context
            
            except Exception as e:
                logging.error(f"Error {e}, using fallback...")
                return self._generate_response_without_memory(query)
        
        else:
            return self._generate_response_without_memory(query=query)