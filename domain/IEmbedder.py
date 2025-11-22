from abc import ABC, abstractmethod
from typing import List

class IEmbedder(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embeds the given text into a Tensor of integers representing the embedding vector.

        Args:
            text (str): The input text to be embedded.

        Returns:
            Tensor[int]: The embedding vector as a Tensor of integers.
        """
        pass
    

    @abstractmethod
    def cleanup_cuda(self) -> None:
        """
        Cleans up CUDA resources if applicable.
        """
        pass


    @abstractmethod
    def shutdown(self) -> None:
        """
        Shuts down any resources used by the embedder.
        """
        pass