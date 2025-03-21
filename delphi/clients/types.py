from typing import Literal, TypedDict, Union


class Message(TypedDict):
    content: str
    role: Literal["system", "user", "assistant"]


ChatFormatRequest = Union[str, list[str], list[Message], None]
