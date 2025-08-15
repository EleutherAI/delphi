"""Neuron simulation scoring components.

This package exposes the unified neuron simulator, scoring helpers, and a
pipeline-compatible scorer wrapper. The API returns standard scoring
structures used elsewhere in the library.
"""

from .types import SimulationResult, AggregateResult, convert_to_legacy_format
from .simulator import NeuronSimulator
from .scoring import simulate_and_score
from .oai_simulator import RefactoredOpenAISimulator

__all__ = [
    "SimulationResult",
    "AggregateResult", 
    "convert_to_legacy_format",
    "NeuronSimulator",
    "simulate_and_score",
    "RefactoredOpenAISimulator",
]
