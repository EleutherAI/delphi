from .client import Client
from .openrouter import OpenRouter
from .server import ServerClient

__all__ = ["Client", "OpenRouter", "Offline", "ServerClient"]


def __getattr__(name):
    if name == "Offline":
        from .offline import Offline

        return Offline
    raise AttributeError(name)
