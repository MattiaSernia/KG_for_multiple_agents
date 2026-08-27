import re
import time
import logging
import ollama
from math import ceil


class TokenCounter:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _get(self, response, key):
        v = getattr(response, key, None)
        if v is None and hasattr(response, "get"):
            v = response.get(key)
        return v or 0

    def add(self, model: str, response):
        self.prompt_tokens += self._get(response, "prompt_eval_count")
        self.completion_tokens += self._get(response, "eval_count")

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


token_counter = TokenCounter()


def ollama_chat(model: str, messages: list, max_retries: int = 3, **kwargs) -> dict:
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model, messages=messages, **kwargs)
            token_counter.add(model, response)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"Ollama call failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)


def uri_to_label(uri) -> str:
    s = str(uri)
    s = s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]
    return s.replace("_", " ")


def sentence_split(text: str, chunk_dim:int) -> list:
    abbrev = r"\b(Mr|Mrs|Dr|Prof|vs|etc|Jr|Sr|Fig|al)\."
    text = re.sub(abbrev, r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences]
    if chunk_dim==0:
        dim=1
    else:
        dim=ceil(len(sentences)*(chunk_dim/100))
    sentences= [" ".join(sentences[i:i+dim]) for i in range(0, len(sentences), dim)]
    return [s for s in sentences if s]
