"""Neuron simulation scoring components.

This package exposes the unified neuron simulator, scoring helpers, and a
pipeline-compatible scorer wrapper. The API returns standard scoring
structures used elsewhere in the library.
"""

from .oai_simulator import RefactoredOpenAISimulator
from .scoring import simulate_and_score
from .simulator import NeuronSimulator
from .types import AggregateResult, SimulationResult, convert_to_legacy_format

__all__ = [
    "SimulationResult",
    "AggregateResult",
    "convert_to_legacy_format",
    "NeuronSimulator",
    "simulate_and_score",
    "RefactoredOpenAISimulator",
]
