
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

class IVectorDB(ABC):
    def __init__(self,
                url: str,
                api_key: Optional[str] = None, 
                collection_name: str = "default_collection",
                dimension: Optional[int] = None,
                distance: Optional[str] = None, 
                top_k: int = 10
                ):   
        
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = distance
        self.top_k = top_k

        """
        Initialize the vector search service.
        :param url: The URL of the vector search service.
        :param api_key: Optional API key for authentication.
        :param collection_name: Name of the collection to use.
        :param dimension: Dimension of the vectors.
        :param distance: Distance metric to use for similarity search.
        :param default_limit: Default number of top results to return.
        """

    @abstractmethod
    async def create_collection(self, recreate: bool) -> None:
        pass

    @abstractmethod
    async def delete_collection(self):
        pass

    @abstractmethod
    async def upsert(self, points: List[Tuple[str, List[float]]], payloads: Optional[List[Dict]] = None) -> None:
        pass

    @abstractmethod
    async def query(self, vector: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def close(self):
        pass