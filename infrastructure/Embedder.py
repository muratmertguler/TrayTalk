
from domain.IEmbedder import IEmbedder
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch

class Embedder(IEmbedder):
    def __init__(self, model_name: str, max_workers: int = 2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)
        self._executor = ThreadPoolExecutor(max_workers, thread_name_prefix="EmbedderThread")
        """FastAPI’s event loop is single-threaded, so running CPU-bound tasks like SentenceTransformers.encode directly blocks other requests. 
            To avoid this, run such tasks in a ThreadPoolExecutor, which keeps the event loop responsive since Torch releases the GIL."""


    async def embed_text(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        try:
            embedding = await loop.run_in_executor(self._executor, 
                                                   lambda: self.model.encode(text, convert_to_tensor=True))
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        
        except Exception as e:
            return f"Embedding failed: {e}"
            

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            print(f"Error during executor shutdown: {e}")


    def cleanup_cuda(self) -> None:
        if self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during CUDA cleanup: {e}")