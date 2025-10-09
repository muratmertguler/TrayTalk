from fastapi import FastAPI
from app.message_router import router as message_router

app = FastAPI()

app.include_router(message_router)

