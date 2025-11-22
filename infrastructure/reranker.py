### Rernaker.py

from domain.IReranker import IReranker
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Reranker(IReranker):
    def __init__(self, model_name: str = None, max_length: int = 1024, device :torch.device = "cpu", workers: int = 2):
        super().__init__(model_name, max_length, workers)
        self.device     = device if device else ("cuda" if torch.cuda.is_available() else "cpu")        
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", trust_remote_code=True)
        self.model      = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device).eval()
        self.model      = self.model.to(self.device) 

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id  = self.tokenizer.convert_tokens_to_ids("yes")

        self.max_length = max_length
        self.prefix     = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        
        self.suffix        =  "<|im_end|>\n<|im_start|>assistant\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)


    def _format(self, query: str, document: str, instruct: Optional[str] = None) -> str:
        instr = instruct or "Provide relevant information based on the Query."
        return f"<Instruct>: {instr}\n<Query>: {query}\n<Document>: {document}"
    

    def _process_input(self, texts: List[str]) -> List[float]:
        max_body = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs   = self.tokenizer(texts, padding=False, truncation="longest_first", 
                                  return_attention_mask=False, max_length=max_body)
        
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", 
                                    max_length=self.max_length)
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.model.device)

        return inputs
    

    @torch.no_grad()
    def _compute_logits(self, intputs) -> List[float]:
        out          = self.model(**intputs)
        batch_score  = out.logits[:, -1, :]
        true_vector  = batch_score[:, self.token_true_id] 
        false_vector = batch_score[:, self.token_false_id]
        pair   = torch.stack([false_vector, true_vector], dim=1)
        logp   = torch.nn.functional.log_softmax(pair, dim=1)
        scores = logp[:, 1].exp().tolist()
        return [float(s) for s in scores]
        

    async def batch(self, query: str, doc: str, instruction: Optional[str] = None) -> float:
        text   = self._format(query, doc, instruction)
        inputs = self._process_input([text])
        scores = self._compute_logits(inputs)
        return float(scores[0]) if scores else 0.0
    

    async def batch_rerank(self, pairs: List[Tuple[str, str]], instruction: Optional[str] = None, batch_size: int = 8) -> List[float]: 
        texts = [self._format(q, d, instruction) for q, d in pairs] 
        results: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self._process_input(batch_texts)
            results.extend(self._compute_logits(inputs))
        return results