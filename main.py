from contextlib import asynccontextmanager
from fastapi import FastAPI
from infrastructure.repository import init_db
from router import book_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔹 Mongo bağlantısı
    client = await init_db()
    print("✅ Beanie initialized.")
    yield   # <-- App burada çalışır
    print("Cleaning up...")
    client.close()

app = FastAPI()
app.include_router(book_router)


@app.get()
def main():
    return {"message": "hello"}