from domain.models import Conversation
from domain.interfaces import IConversationRepository
from beanie import Document
from datetime import datetime


class ConversationDocument(Document):
    session_id: str
    user_id: int
    dict_conversation: dict[str, str]
    timestamp: datetime

    class Collection:
        name = "conversations"
        indexes = [
            "session_id",  
            "user_id",
            "timestamp"
        ]

    def to_domain(self) -> Conversation:
        return Conversation(
            session_id=self.session_id,
            user_id=self.user_id,
            dict_conversation=self.dict_conversation,
            timestamp=self.timestamp
        )

    @staticmethod
    def from_domain(conversation: Conversation) -> "ConversationDocument":
        return ConversationDocument(
            session_id=conversation.session_id,
            user_id=conversation.user_id,
            dict_conversation=conversation.dict_conversation,
            timestamp=conversation.timestamp
        )


class ConversationRepository(IConversationRepository):
    async def save_conversation(self, conversation: Conversation):
        conversation_doc = ConversationDocument.from_domain(conversation)
        await conversation_doc.insert()

    async def get_conversation(self, session_id: str) -> Conversation | None:
        conversation_doc = await ConversationDocument.find_one(Conversation.session_id == session_id)
        return conversation_doc.to_domain() if conversation_doc else None
