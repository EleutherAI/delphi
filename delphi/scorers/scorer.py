from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional

from ..latents.latents import LatentRecord


class ScorerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through."""

    score: Any
    """Generated score for latent."""

    duration: Optional[float] = None
    """Time taken to generate the score in seconds."""


class Scorer(ABC):
    @abstractmethod
    async def __call__(self, record: LatentRecord) -> ScorerResult:
        pass
