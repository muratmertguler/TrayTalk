from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class User(BaseModel):
    id: int = Field(..., description="Unique identifier for the user")
    name :str = Field(..., description="Name of the user")


class Message(BaseModel):
    user_id : int = Field(..., description="ID of the user who sent the message")
    message_id =    Field(..., description="Unique identifier for the message")
    content : str = Field(..., description="Content of the message")
    system_prompt : str | None = Field(default=None, description="System prompt associated with the message, if any")
    timestamp : datetime =  Field(default_factory=datetime.now, description="Timestamp of when the message was sent")

    def to_json(self):
        return self.model_dump_json()


class Response(BaseModel):
    model_name: str  = Field(..., description="Name of the model that generated the response")
    response_id: int = Field(..., description="Unique identifier for the response")
    content: str =     Field(..., description="Content of the response")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp of when the response was generated")

    def to_dict(self):
        return  self.model_dump_json()
    

class Conversation(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the conversation session")
    user_id: int = Field(..., description="User who owns the conversation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Conversation start time")
    interactions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of message-response interactions"
    )

    def add_interaction(self, user_message: str, system_prompt: str 
                        ,ai_response: str, metadata: Dict = None):
        interaction = {
            "user_message": user_message,
            "ai_response": ai_response,
            "system_prompt": system_prompt,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.interactions.append(interaction)
        self.updated_at = datetime.now()

    def get_conversation_history(self) -> List[Dict]:
        return self.interactions

    def get_last_n_interactions(self, n: int) -> List[Dict]:
        return self.interactions[-n:]