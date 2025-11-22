from abc import abstractmethod
from typing import Optional

class  ILLMClient:
    def __init__(self, model_name: str, url: str, temperature: float = 0.2, 
                 max_input_tokens: int = 2048, max_output_tokens: int = 1024,
                 api_key: Optional[str] = None):
        
        self.model_name = model_name
        self.url = url
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """
        Generates text based on the given prompt.

        Args:
            prompt (str): The input prompt to generate text from.

        Returns:
            str: The generated text.
        """
        pass        


    @abstractmethod
    async def generate_text_stream(self, prompt: str):
        """
        Generates text based on the given prompt as a stream.

        Args:
            prompt (str): The input prompt to generate text from.

        Yields:
            str: The generated text stream.
        """     
        pass
    

    @abstractmethod
    async def enrich_question(self, prompt: str, text: str) -> str:
        """
        Enriches the given text based on the provided prompt.

        Args:
            prompt (str): The input prompt to guide the enrichment.
            text   (str): The text to be enriched.

        Returns:
            str: The enriched text.
        """
        pass