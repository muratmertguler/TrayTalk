from fastapi import FastAPI, Depends, HTTPException
from llm_services.core.llm_base import LLMClient, LLMRequest, LLMResponse
from llm_services.core.llm_clients import OllamaClient
from database import ConversationHistory
import uvicorn
import subprocess
import socket
from datetime import datetime
from uuid import uuid4

app = FastAPI()

conversation_history = ConversationHistory()
session_id = str(uuid4())

try:
    hostname = socket.gethostbyname()
except:
    try:
        result = subprocess.run(["hostname"], capture_output=True, text=True)
        hostname = result.stdout.split() if result.returncode == 0 else "default"
    except:
        hostname = "default"


def get_llm_client() -> LLMClient:
    return OllamaClient()


@app.post("/chat", response_model=LLMResponse)
async def chat(request:LLMRequest, llm:LLMClient = Depends(get_llm_client)):
    try:
        answer = llm.generate_response(request.query, session_id=session_id)
        conversation_history.save_conversation(
            user_id=hostname,
            session_id=request.session_id,
            llm_name=llm.llm_name,
            user_message=request.query,
            llm_response=answer,
            system_prompt=llm.system_prompt if hasattr(llm, "system_prompt") else "",
            timestamp=datetime.now()
        )

        return LLMResponse(query=request.query, answer=answer, session_id=session_id, 
                           timestamp=datetime.now(), success=True, model_used=llm.get_llm_name())
    
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Error Processing Request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
