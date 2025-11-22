    
from typing import AsyncIterator
from fastapi.responses import StreamingResponse
from domain.ILLMClient import ILLMClient
from services.RerankerService import RerankerService

class LLMChatService:
    def __init__(self, llm_client: ILLMClient, reranker_service: RerankerService):
        self.llm_client = llm_client
        self.reranker_service = reranker_service
        
    async def stream_response(self, query: str, system_prompt: str = None) -> StreamingResponse:
        rag_info = self.reranker_service.rerank_documents(query)
        
        async def generate():
            try:
                async for chunk in self.llm_client.generate_text_stream(query, system_prompt, rag_info):
                    yield f"data: {chunk}\n\n"
                yield "data: [END]\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")