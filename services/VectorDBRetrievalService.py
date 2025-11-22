
from typing import List, Dict
from domain.IVectorDB import IVectorDB
from domain.IEmbedder import IEmbedder
import uuid


class VectorDBManagerService:
    def __init__(self, db: IVectorDB, embedder: IEmbedder):
        self.db = db
        self.embedder  = embedder

    async def upsert(self, texts: List[str] | List[dict], metadata: List[Dict]) -> List[str]:
        """
        Upsert texts into the vector database after embedding them.
        """
        embeddings = []
        for text in texts:
            embedding = await self.embedder.embed_text(text)
            embeddings.append(embedding)

        points = []
        ids = []
        for i in range(len(texts)):
            point_id = str(uuid.uuid4())
            points.append((point_id, embeddings[i]))
            ids.append(point_id)

        await self.db.upsert(points, payloads=metadata)
        return ids


    async def batch_upsert(self, list_text: List[dict]) -> List[str]:
        data = [text.dict() for text in list_text]
    
        texts = [item["content"] for item in data]
        
        embeddings = await self.embedder.embed_text(texts)
        
        points = []
        ids = []
        payloads = []
        
        for i, item in enumerate(data):
            point_id = str(uuid.uuid4())
            points.append((point_id, embeddings[i]))
            ids.append(point_id)
            
            payload = {
                "content": item["content"],
                "tags": item.get("tags", []),
                "lang": item.get("lang", "en"),
                "created_at": item.get("created_at").isoformat(),
            }
            payloads.append(payload)
        
        # Batch upsert all points
        await self.db.upsert(points, payloads=payloads)
        return ids


    async def search(self, query: str, top_k: int) -> List[Dict]:
        """
        Search for similar vectors in the vector database based on the input query.
        """
        embedding = await self.embedder.embed_text(query)
        return await self.db.query(embedding, top_k)
    