"""
Scoring utilities for simulator outputs.

This module provides asynchronous helpers to run simulations over multiple
sequences, compute per-sequence correlations, and aggregate results by group
(e.g., quantile) using correlation over concatenated activations.
"""

import asyncio
import logging
from collections import defaultdict
from typing import List, Optional

import numpy as np

from .data_models import ActivationRecord
from .simulator import NeuronSimulator
from .types import (
    AggregateResult,
    ScoredSimulation,
    SequenceSimulation,
    SimulationResult,
    convert_to_legacy_format,
)

logger = logging.getLogger(__name__)


# Utility functions for backward compatibility with tests
def correlation_score(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate correlation score between true and predicted values."""
    return _calculate_correlation(y_true, y_pred)


def fix_nan(val: float) -> str | float:
    """Convert numpy NaN to string "nan" for JSON serialization."""
    if np.isnan(val):
        return "nan"
    else:
        return float(val)


def score_from_simulation(
    activation_record: ActivationRecord,
    sequence_simulation: SequenceSimulation,
    correlation_fn,
) -> float:
    """Calculate correlation score from a simulation result (backward compatibility)."""
    return correlation_fn(
        activation_record.activations, sequence_simulation.expected_activations
    )


async def simulate_and_score(
    simulator: NeuronSimulator,
    activation_records: List[ActivationRecord],
    non_activation_records: Optional[List[ActivationRecord]] = None,
) -> List[ScoredSimulation]:
    """Run simulation and compute correlations for multiple sequences.

    Executes simulations concurrently for the provided activation records,
    computes per-sequence Pearson correlations, groups results by quantile, and
    aggregates correlations over concatenated activations. When
    ``non_activation_records`` are provided, an overall group (quantile ``-1``)
    is added that combines all sequences.

    Args:
        simulator: Simulator used to predict token-level activations.
        activation_records: Sequences with true activations to score.
        non_activation_records: Optional sequences expected to be non-activating.

    Returns:
        A list of ``AggregateResult`` instances, one per quantile group, plus an
        optional overall group when ``non_activation_records`` are provided.
    """
    if non_activation_records is None:
        non_activation_records = []

    async def _simulate_and_score_record(record):
        predicted = await simulator.simulate(record.tokens)

        # Handle both SequenceSimulation objects and legacy list returns
        if hasattr(predicted, "expected_activations"):
            predicted_activations_list = list(predicted.expected_activations)
        elif isinstance(predicted, list):
            predicted_activations_list = predicted
        else:
            # Other iterable type - shouldn't happen
            predicted_activations_list = list(predicted)  # type: ignore

        correlation = _calculate_correlation(
            record.activations, predicted_activations_list
        )
        return SimulationResult(
            tokens=record.tokens,
            predicted_activations=predicted_activations_list,
            true_activations=record.activations,
            correlation=correlation,
            quantile=record.quantile,
        )

    results = await asyncio.gather(
        *[_simulate_and_score_record(record) for record in activation_records]
    )

    groups = defaultdict(list)
    for result in results:
        groups[result.quantile].append(result)

    if non_activation_records:
        non_activating_results = await asyncio.gather(
            *[_simulate_and_score_record(record) for record in non_activation_records]
        )

        all_sequences = results + non_activating_results

        # Add overall group at quantile -1
        groups[-1] = all_sequences

    # Return list of aggregate results converted to legacy format
    aggregates = [
        _aggregate_group(quantile, sequences) for quantile, sequences in groups.items()
    ]
    return convert_to_legacy_format(aggregates)


def _aggregate_group(
    quantile: int, sequences: List[SimulationResult]
) -> AggregateResult:
    """Aggregate per-sequence results into a group-level metric.

    Correlation is computed over the concatenation of all true and predicted
    activations from ``sequences``. This is not an average of per-sequence
    correlations.

    Args:
        quantile: Group identifier for the aggregate.
        sequences: Per-sequence results to aggregate.

    Returns:
        ``AggregateResult`` with group-level correlation and metadata.
    """
    if not sequences:
        return AggregateResult(
            quantile=quantile, correlation=0.0, sequence_count=0, sequences=[]
        )

    all_true = []
    all_predicted = []
    for seq in sequences:
        all_true.extend(seq.true_activations)
        all_predicted.extend(seq.predicted_activations)

    correlation = _calculate_correlation(all_true, all_predicted)

    return AggregateResult(
        quantile=quantile,
        correlation=correlation,
        sequence_count=len(sequences),
        sequences=sequences,
    )


def _calculate_correlation(
    true_activations: List[float], predicted_activations: List[float]
) -> float:
    """Compute Pearson correlation between two aligned sequences.

    This function exactly matches the original correlation_score implementation
    to ensure identical behavior for logprob-free scoring.

    Args:
        true_activations: Ground-truth activation values per token.
        predicted_activations: Predicted activation values per token.

    Returns:
        Pearson correlation coefficient as a float (may be NaN for degenerate cases).
    """
    return np.corrcoef(true_activations, predicted_activations)[0, 1]
