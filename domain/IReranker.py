from typing import List, Optional, Tuple


class IReranker:
    def __init__(self, model_name: str = None, max_length: int = 1024, device=None, workers: int = 2):
        self.model_name = model_name
        self.max_length = max_length
        self.device     = device
        self.workers    = workers
        

    async def batch(self, query: str, doc: str, instruction: Optional[str] = None) -> float:
        """Rerank the given contexts based on their relevance to the query.
        Args:
            query (str): The input query string.
            contexts (List[str]): A list of context strings to be reranked.
        Returns:
            List[Tuple[str, float]]: A list of tuples containing the context and its relevance score.
        """
        pass

    async def batch_rerank(self, pairs: List[Tuple[str, str]], instruction: Optional[str] = None, batch_size: int = 8) -> List[float]: 
        """Rerank the given contexts for each query in the batch.
        Args:
            queries (List[str]): A list of input query strings.
            contexts_list (List[List[str]]): A list of lists, where each sublist contains context strings to be reranked for the corresponding query.
        Returns:
            List[List[Tuple[str, float]]]: A list of lists, where each sublist contains tuples of context and its relevance score for the corresponding query.
        """
        pass