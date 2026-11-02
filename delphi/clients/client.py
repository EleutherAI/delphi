from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class Response:
    text: str
    logprobs: list[Any] | None = None
    prompt_logprobs: list[Any] | None = None


class Client(ABC):
    supports_prompt_logprobs = False

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(
        self, prompt: Union[str, list[dict[str, str]]], **kwargs
    ) -> str | Response:
        pass

    # @abstractmethod
    # async def process_response(self, raw_response: Any) -> Response:
    #     pass
