from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from domain.models import Conversation
from domain.interfaces import IMessageRepository

async def init_db(mongo_uri: str, db_name: str = "gardiyan_db"):
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    await init_beanie(database=db, document_models=[Conversation])
    return db

class MongoRepository(IMessageRepository):
    def __init__(self, db):
        self.db = db

    async def save_message(self, message: Conversation) -> None:
        await self.db["messages"].insert_one(message.dict())

    async def get_conversation(self, session_id: str):
        cursor = self.db["messages"].find({"session_id": session_id}).sort("timestamp", 1)
        return [Conversation(**doc) async for doc in cursor]