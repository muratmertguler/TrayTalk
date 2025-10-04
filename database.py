from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Dict, Any, List

class ConversationHistory:
    def __init__(self, 
                url:str = "mongodb://admin:1@localhost:27017",
                db_name:str = "TrayTalk_db",
                collection_name = "conversations"
                ):
        self.client = MongoClient(url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.collection.create_index([("session_id", 1), ("timestamp", -1)])
        self.collection.create_index([("timestamp", -1)])

    def save_conversation(self, user_id:str, session_id:str, llm_name:str, user_message:str, 
                          llm_response:str,  system_prompt:str, timestamp:datetime=None):

        record = {
            "user_id":user_id,
            "session_id":session_id,
            "llm_name": llm_name,
            "user_message":user_message,
            "llm_response":llm_response,
            "system_prompt":system_prompt,
            "timestamp":timestamp or datetime.now()
        }
        self.collection.insert_one(record)

    def get_user_conversations(self, user_id:str="default", limit:int=10) -> List[Dict[str, Any]]:
        records = self.collection.find({"user_id":user_id}).sort("timestamp", -1).limit(limit)
        return list(records)
    
    def get_session_conversations(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        records = self.collection.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
        return list(records)
    
    def close_connection(self):
        """Close Connection"""
        self.client.close()
