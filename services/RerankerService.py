from services.VectorDBRetrievalService import VectorDBManagerService
from domain.IReranker import IReranker

class RerankerService:
    def __init__(self, db: VectorDBManagerService, reranker: IReranker):
        self.db = db
        self.reranker = reranker

    async def rerank_documents(self, query: str, threshold: float = 0.0) -> list[dict]:
        """
        Rerank documents based on their relevance to the input query.
        """
        retrieved_docs = await self.db.search(query, top_k=10)
        valid_docs = []
        pairs = []
        for doc in retrieved_docs.points:
            payload = getattr(doc, "payload", {}) or {}
            content = payload.get("content")
            
            if content:
                valid_docs.append(doc)
                pairs.append((query, content))

        if not pairs:
            return []

        scores = await self.reranker.batch_rerank(pairs)
        ranked_results = []
        for doc, score in zip(valid_docs, scores):
            # Threshold kontrolü (isteğe bağlı, çok düşük skorları elemek için)
            if score < threshold:
                continue

            payload = getattr(doc, "payload", {})
            
            ranked_results.append({
                "content": payload.get("content"),
                "score": float(score), # Numpy float gelirse native float'a çevir
                "id": str(doc.id),
                "metadata": {k: v for k, v in payload.items() if k != "content"} # Content hariç diğer bilgiler
            })

        ranked_results.sort(key=lambda x: x["score"], reverse=True)

        return ranked_results
