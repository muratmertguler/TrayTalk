
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

__all__ = [
    "UserMessageRequest",
    "LLMMessageResponse",
    "BatchUpsertRequest",
    "UpsertRequest"
          ]

## USER Message Schema

class UserMessageRequest(BaseModel):
    user_id    : Optional[str] = Field(description="Unique identifier for the user")
    session_id : Optional[str] = Field(description="Unique identifier for the session")
    content    : str = Field(description="The message content sent by the user")
    timestamp  : Optional[datetime] = Field(
        default_factory=lambda: datetime.now(utctime=True),
        description="Timestamp of when the message was sent in ISO 8601 format"
    )


## LLM Response Schema

class LLMResponseMessage(BaseModel):
    role   : str = Field(description="The role of the message sender, e.g., 'assistant'")
    content: str = Field(description="The content of the message")

class LLMMessageResponse(BaseModel):
    model             : str = Field(description="The model used for generating the response")
    created_at        : datetime = Field(description="Timestamp when the response was created")
    message           : LLMResponseMessage  = Field(description="The message content from the LLM")
    done_reason       : Optional[str] = Field(description="Reason why the response generation was completed")
    done              : bool = Field(description="Indicates if the response generation is complete")
    total_duration    : int  = Field(description="Total duration taken for generating the response in nanoseconds")
    load_duration     : int  = Field(description="Duration taken to load the model in nanoseconds")
    prompt_eval_count : int  = Field(description="Number of prompt evaluations performed")
    eval_count        : int  = Field(description="Number of tokens generated in the response") 
    eval_duration     : int  = Field(description="Total duration of evaluations in nanoseconds")

## Vector Response Schema

class VectorDBRetrivalContext(BaseModel):
    id       : str = Field(description="Unique identifier for the retrival context")
    contents : str = Field(description="The content of the retrival context")
    metadata : Optional[dict] = Field(description="Additional metadata associated with the context")
    score    : Optional[float] = Field(description="Relevance score of the context")


## Batch Upsert 

class BatchMessageRequest(BaseModel):
    user_id: Optional[str] = Field(description="Unique identifier for the user")
    session_id: Optional[str] = Field(description="Unique identifier for the session")
    content: str = Field(description="The message content sent by the user")
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(utctime=True),
        description="Timestamp of when the message was sent in ISO 8601 format"
    )

class BatchUpsertRequest(BaseModel):
    messages: List[BatchMessageRequest] = Field(description="List of messages to upsert")

class UpsertRequest(BaseModel):
    content: str = Field(description="The message content from the docs")
    tags  : str = Field(description="Tags")
    lang  : Optional[str] = Field("en", description="langue") 
    created_at: Optional[datetime] = Field(default_factory=datetime.now())
