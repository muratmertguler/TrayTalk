import os
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

from domain.interfaces import ILLMClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class OllamaClient(ILLMClient):
    def __init__(self, base_url: str = None, model: str = None):
        base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        model = model or os.getenv("LLM_MODEL", "llama3.2:latest")
        self.llm = ChatOllama(model=model, base_url=base_url)
        logging.info(f"Using Ollama model: {model} at {base_url}")
        self.system_prompt = "you are a assistance"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history")
        ])

        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def response_generation(self, query:str, session_id:str = "default") -> str:
        self.history = self.get_session_history(session_id)
        if session_id not in self.memory_store:
            try:
                response = self.chain_with_history.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": session_id}}
                )
                return response
            
            except Exception as e:
                print(f"Error : {e}, using fallback...")
                return self._response_generation_without_memory(query)
            
        else:
            return self._response_generation_without_memory(query)
