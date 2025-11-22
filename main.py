from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.router import message_router
from contextlib import asynccontextmanager
import signal
import sys

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("Starting up the application...")
    yield
    # Shutdown actions
    print("Shutting down the application...")

app = FastAPI(
    title="Chatbot API",
    description="A simple chatbot API using FastAPI",
    version="1.0.0",
    lifespan=lifespan 
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(message_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API!"}

def signal_handler(sig, frame):
    print("Signal received, shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  
signal.signal(signal.SIGTERM, signal_handler) 