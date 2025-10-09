from domain.interfaces import IConversationRepository
from domain.interfaces import ILLMClient
from domain.models import Message

class MessageService:
    def __init__(self, conversation_repository: IConversationRepository, llm_client: ILLMClient):
        self.conversation_repository = conversation_repository
        self.llm_client = llm_client 

    async def handle_message(self, message: Message):
        prompt = f"User says: {message.content}"
        llm_response = await self.llm_client.generate_response(prompt)
        conversation = await self.conversation_repository.save(user_id=message.user_id, message=message.content, response=llm_response)
        conversation.add_message(message)
        await self.conversation_repository.save(conversation=conversation)
        return llm_response
