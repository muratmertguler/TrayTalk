import asyncio
from infrastructure.generation.OllamaClient import OllamaClient  # sınıfının olduğu dosya adıyla değiştir
from infrastructure.memory import MemoryManager


memory = MemoryManager()
client = OllamaClient(model="llama3.2:3b", base_url="http://localhost:11434", memory_manager=memory)
SESSION_ID = "default"

messages = [
        {"role": "system", "content": "you are a assitance."},
        {"role": "user", "content": "Hello, how are you ?"}
    ]



async def main():

    async for response in client.response_generation(messages):
        print(response, end="", flush=True)
    
    print("\n")

asyncio.run(main())

