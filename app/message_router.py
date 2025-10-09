from fastapi import APIRouter, Depends
from app.dependencies import get_current_user

router = APIRouter()

@router.post("/message")
def send_message(message: str, user: dict = Depends(get_current_user)):
    return {"user": user["username"], "message": message}
