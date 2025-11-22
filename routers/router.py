from fastapi import APIRouter, Depends
from domain.ILLMMessageHandler import ILLMMessageHandler
from domain.IUserMessageHandler import IUserMessageHandler
from domain.IEmbedder import IEmbedder
from domain.IReranker import IReranker
from services.VectorDBRetrievalService import VectorDBManagerService
from routers.schemas import *
from services.RerankerService import RerankerService
from routers.dependencies import *
from typing import List

message_router = APIRouter(
    prefix="/messages",
    tags=["messages"],
    responses={ 200: {"description": "Successful Response"},
                404: {"description": "Not found"},
                500: {"description": "Internal server error"}},
    )


@message_router.get("/healthCheck", response_model=dict)
async def health_check() -> dict:
    """
    Health check endpoint to verify that the message router is operational.
    """
    return {"status": "Message router is operational"}


#----------------------------#
#-Message-Handling-Endpoints-#
#----------------------------#

@message_router.post("/user/sendMessage", response_model=dict)
async def send_message(
    request: UserMessageRequest,
    service: IUserMessageHandler = Depends(get_user_message_handler_service)
    ):

    return await service.message_handler(request.user_id, 
                                                      request.session_id, 
                                                      {"content":request.content}, 
                                                      request.timestamp) 


@message_router.post("/llm/sendMessage", response_model=dict)
async def response_llm_message_handler(
    request: LLMMessageResponse,
    service: ILLMMessageHandler = Depends(get_llm_message_handler_service)
    ): 
    
    return await service.message_handler(request.model_dump())


#----------------------------#
#----------Embedder----------#
#----------------------------#

@message_router.post("/llm/embedder", response_model=dict)
async def embedder_message_handler(
    request: UserMessageRequest,
    service: IEmbedder = Depends(get_embedder)
    ): 

    return await service.embed_text(request.content)


#----------------------------#
#----Vector-DB-Operations----#
#----------------------------#
@message_router.get("/vectordbOperation/createDb", response_model=dict)
async def vectordb_create_handler(
    service: VectorDBManagerService = Depends(get_vector_manager_service)
    ):
    
    await service.db.create_collection()
    return {"status": "collection created"}


@message_router.post("/vectordbOperation/upsert", response_model=dict)
async def vectordb_upsert_handler(
    request: UpsertRequest,
    service: VectorDBManagerService = Depends(get_vector_manager_service)
    ):
    
    result = await service.upsert([request.content], [{"user_id": request.user_id, "session_id": request.session_id, "content": request.content}])
    return {"status": "success",
        "upserted_ids": result}


@message_router.post("/vectordbOperation/upsertBatch", response_model=dict)
async def vectordb_upsert_batch_handler(
    request: List[UpsertRequest],
    service: VectorDBManagerService = Depends(get_vector_manager_service)
):
    result = await service.batch_upsert(request)
    return {
        "status": "success",
        "upserted_ids": result
    }
    

@message_router.post("/vectordbOperation/search", response_model=dict)
async def vectordb_search_handler(
    request: UserMessageRequest,
    service: VectorDBManagerService = Depends(get_vector_manager_service)
    ):
    
    result = await service.search(request.content, top_k=5)
    return {"results": result}


#----------------------------#
#-----------Reranke----------#
#----------------------------#
@message_router.post("/rerank", response_model=dict)
async def rerank_handler(
    request       : UserMessageRequest,
    service       : RerankerService = Depends(get_reranker_service)
    ): 
    
    result = await service.rerank_documents(request.content)
    return {"results": result}
