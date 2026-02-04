from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    @abstractmethod
    def generate(prompts: List[str], **kwargs) -> List[str]
    @abstractmethod
    def batch_generate(prompts: List[str], batch_size: int) -> List[str]
    def get_token_count(text: str) -> int