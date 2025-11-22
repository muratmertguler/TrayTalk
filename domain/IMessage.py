from datetime import datetime
from beanie import Document, Indexed
from pydantic import Field
from typing import Optional, List, Dict, Any

class UserMessage(Document):
    user_id    : Indexed[str] = Field(..., description="Unique identifier for the user")
    session_id : Indexed[str] = Field(..., description="Unique identifier for the session")
    message    : str = Field(..., description="The content of the user's message")
    timestamp  : datetime = Field(default_factory=datetime.utcnow, description="Timestamp of when the message was created")

class LLMMessage(Document):
    Model   : str = Field(..., description="The model used for generating the message")
    role    : str = Field(..., description="Role of the message sender (e.g., user, assistant)")
    content : str = Field(..., description="The content of the LLM message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the message")
