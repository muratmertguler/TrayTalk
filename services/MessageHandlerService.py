########## services ##########

from domain.ILLMMessageHandler import ILLMMessageHandler
from domain.IUserMessageHandler import IUserMessageHandler
from typing import Any
from datetime import datetime



class UserMessageHandlerService(IUserMessageHandler):
    async def message_handler(
            self, user_id: str, 
            session_id   : str, 
            message      : dict, 
            timestamp    : datetime
            ) -> dict[str, Any]:
        
            try:
                if not user_id or not session_id or not message:
                    raise ValueError("Invalid input data")
                
                message = message.get("content", "")
                if not message:
                    raise ValueError("Message content is empty")
            
                response = {
                "user_id"   : user_id,
                "session_id": session_id,
                "message"   : message,
                "timestamp" : timestamp.isoformat(timespec='seconds')}
            
                return response
            
            except Exception as e:
                print(f"Error processing message: {e}")
                return None



class LLMMessageHandlerService(ILLMMessageHandler):
    async def message_handler(self, message: dict) -> dict[str, Any]:
        try:
            if not message:
                raise ValueError("LLM message is empty")
            
            message_content = message.get("message", {}).get("content", "")
            if not message:
                raise ValueError("LLM message content is empty")
            
            response = {
                "model": message.get("model", ""),
                "created_at": message.get("created_at", ""),
                "message": {
                    "role": message.get("message", {}).get("role", ""),
                    "content": message_content
                },
                "done_reason"   : message.get("done_reason", ""),
                "done"          : message.get("done", False),
                "total_duration": message.get("total_duration", 0),
                "load_duration" : message.get("load_duration", 0),
                "prompt_eval_count": message.get("prompt_eval_count", 0),
                "eval_count"    : message.get("eval_count", 0),
                "eval_duration" : message.get("eval_duration", 0)
            }

            return response

        except Exception as e:
            print(f"Error processing LLM message: {e}")
            return False, {}
        

class VectorRetrivalContextHandlerService:
    async def message_handler(self, retrieval_results: dict) -> dict[str, Any]:
        try:
            if not retrieval_results or not isinstance(retrieval_results, dict):
                raise ValueError("Invalid retrieval results")

            points = retrieval_results.get("results", {}).get("points", [])
            if not isinstance(points, list):
                raise ValueError("Invalid format for points")

            items = []
            for p in points:
                if not isinstance(p, dict):
                    continue
                payload = p.get("payload", {}) or {}
                content = payload.get("content")
                items.append({
                    "id": p.get("id", ""),
                    "score": p.get("score", 0.0),
                    "content": content
                })

            context = " ".join([it["content"] for it in items if it["content"]])
            return {"count": len(items), "items": items, "context": context}

        except Exception as e:
            print(f"Error processing vector retrieval results: {e}")
            return None