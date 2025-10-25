from datetime import datetime
from beanie import Document, Indexed
from pydantic import Field
from typing import Optional, List, Dict, Any

class Conversation(Document):
    user_id: int = Indexed(int)
    session_id: str = Indexed(str)
    role: Optional[str] = Field(None, description="system, user, assistant")
    user_message: Optional[str] = Field(None, description="The content of the user message")
    system_prompt: Optional[str] = Field(None, description="The system prompt used for this message")
    ai_response: Optional[str] = Field(None, description="The AI's response to the user message")
    model_name: Optional[str] = Field(None, description="Model name used for generating the response")
    temperature: Optional[float] = Field(None, description="Temperature setting for the model")
    max_input_tokens: Optional[int] = Field(None, description="Max tokens setting for the model")
    max_response_tokens: Optional[int] = Field(None, description="Max response tokens setting for the model")
    timestamp: Optional[float] = Field(default_factory=lambda: datetime.now().timestamp(), description="Unix timestamp of the message")
    interactions: List[Dict[str, Any]] = Field(default_factory=list, description="List of interactions")

    def add_interaction(
        self,
        user_message: str,
        system_prompt: str,
        ai_response: str,
        metadata: Dict = None
    ):
        interaction = {
            "user_message": user_message,
            "system_prompt": system_prompt,
            "ai_response": ai_response,
            "timestamp": datetime.now().timestamp(),
            "metadata": metadata or {}
        }
        self.interactions.append(interaction)