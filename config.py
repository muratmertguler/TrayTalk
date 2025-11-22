from pydantic_settings import BaseSettings
from typing import Optional

class ConfigEmbedder(BaseSettings):
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    
class ConfigQdrantDB(BaseSettings):
    url: str = "http://localhost:6333"
    api_key        : Optional[str] = None
    collection_name: str = "chatbot"
    dimension : int = 1024
    distance  : str = "Cosine"
    top_k     : int = 10

class ConfigReranker(BaseSettings):
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 1024
    workers   : int = 2