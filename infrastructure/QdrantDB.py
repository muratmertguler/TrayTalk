
from uuid import uuid4
from typing import List, Tuple, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from domain.IVectorDB import IVectorDB

from concurrent.futures import ThreadPoolExecutor
import asyncio

class QdrantDB(IVectorDB):
    def __init__(self,
                 url: Optional[str] = None,
                 port: Optional[int] = 6333,
                 collection_name: str = None, 
                 dimension: int = None,
                 distance = None, 
                 top_k = 10,
                 executor: Optional[ThreadPoolExecutor] = None):
        
        self.url = url if url else "http://localhost:6333"
        self.client = QdrantClient(url=self.url, port=port)
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = distance if distance else models.Distance.COSINE
        self.top_k = top_k
        self._executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="QdrantdbWorker")


    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))


    def _ensure_collection(self):
        if self.client.collection_exists(collection_name=self.collection_name):
            return True
        else:
            return False


    async def create_collection(self):    
        exists = await self._run_sync(self.client.collection_exists, collection_name=self.collection_name)
        
        if not exists:
            await self._run_sync(
                self.client.create_collection,
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)
            )
        else:
            raise ValueError(f"Collection {self.collection_name} already exists.")


    async def delete_collection(self):
        exists = await self._run_sync(self.client.collection_exists, collection_name=self.collection_name)
        
        if exists:
            await self._run_sync(self.client.delete_collection, collection_name=self.collection_name)
        else:
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        

    async def upsert(self, points: List[Tuple[str, List[float]]], payloads: Optional[List[Dict]] = None) -> None:
        if self._ensure_collection():
            
            point_structs = []
            for i, (id_, vec) in enumerate(points):
                payload = payloads[i] if payloads and i < len(payloads) else {}
                point_structs.append(models.PointStruct(id=id_, vector=vec, payload=payload))
            try:
                await self._run_sync(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=point_structs
                )
            except Exception:
                print("upsert failed")

        else:
            print(f"Collection {self.collection_name} does not exist.")
            return []


    async def query(self, point: List[float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._ensure_collection():
            k = top_k if top_k is not None else self.top_k
            results = await self._run_sync(
                self.client.query_points,
                collection_name=self.collection_name,
                query=point,
                limit=k
            )
            return results
            
        else:
            print(f"Collection {self.collection_name} does not exist.")
            return []


    async def get_collection_info(self) -> Dict:
        if self._ensure_collection():
            info = await self._run_sync(
                self.client.get_collection,
                collection_name=self.collection_name
            )
            return info.dict()
        else:
            print(f"Collection {self.collection_name} does not exist.")
            return {}
        
        
    async def delete_vectors(self, ids: List[str]) -> None:
        if self._ensure_collection():
            await self._run_sync(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=models.PointsSelector(ids=ids)
            )
        else:
            print(f"Collection {self.collection_name} does not exist.")
            return None
    
    async def close(self):
        await self._run_sync(self.client.close)