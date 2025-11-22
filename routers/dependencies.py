### dependency.py

from functools import lru_cache
from services.MessageHandlerService import UserMessageHandlerService, LLMMessageHandlerService
from services.RerankerService import RerankerService
from infrastructure.Embedder import Embedder
from infrastructure.QdrantDB import QdrantDB
from infrastructure.Reranker import Reranker
from services.VectorDBRetrievalService import VectorDBManagerService
from config import ConfigEmbedder, ConfigQdrantDB, ConfigReranker


__all__ = [
    "get_user_message_handler_service",
    "get_llm_message_handler_service",
    "get_vector_manager_service",
    "get_reranker_service",
    "get_embedder",
    "get_vectordb",
    "get_reranker",
]


@lru_cache()
def get_user_message_handler_service() -> UserMessageHandlerService:
    return UserMessageHandlerService()


@lru_cache()
def get_llm_message_handler_service() -> LLMMessageHandlerService:
    return LLMMessageHandlerService()


@lru_cache()
def get_embedder() -> Embedder:
    config = ConfigEmbedder()
    return Embedder(config.model_name)


@lru_cache()
def get_vectordb() -> QdrantDB:
    config = ConfigQdrantDB()
    # pass args by name to match QdrantDB ctor; keep DI in infrastructure
    return QdrantDB(url=config.url, collection_name=config.collection_name,
                    dimension=config.dimension, distance=config.distance,
                    top_k=config.top_k)


@lru_cache()
def get_vector_manager_service() -> VectorDBManagerService:
    # small factory for VectorManagerService: uses interfaces returned by other providers
    return VectorDBManagerService(db=get_vectordb(), embedder=get_embedder())


@lru_cache()
def get_reranker() -> Reranker:
    config = ConfigReranker()
    return Reranker(model_name=config.model_name,
                    max_length=config.max_length,
                    workers=config.workers)


@lru_cache()
def get_reranker_service() -> RerankerService:
    return RerankerService(db=get_vector_manager_service(), reranker=get_reranker())