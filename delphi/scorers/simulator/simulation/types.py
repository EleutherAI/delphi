"""Complete type system for simulator operations.

This module contains:
- Legacy compatibility types: SequenceSimulation, ScoredSimulation, etc.
- Modern clean types: SimulationResult, AggregateResult
- Conversion functions between type systems
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from simple_parsing import Serializable

# === LEGACY TYPES (for backward compatibility) ===


class ActivationScale(str, Enum):
    """
    Which "units" are stored in the expected_activations/distribution_values fields of
    a SequenceSimulation.
    """

    SIMULATED_NORMALIZED_ACTIVATIONS = "simulated_normalized_activations"
    SIMULATED_UNNORMALIZED_ACTIVATIONS = "simulated_unnormalized_activations"


@dataclass
class SequenceSimulation(Serializable):
    """The result of a simulation of neuron activations on one text sequence."""

    tokens: list[str]
    """The sequence of tokens that was simulated."""

    expected_activations: list[float]
    """The expected activation value for each token."""

    activation_scale: Optional[ActivationScale]
    """The scale/units of the expected_activations values."""

    distribution_values: Optional[list[list[int]]] = None
    """For each token, the possible activation values in the distribution."""

    distribution_probabilities: Optional[list[list[float]]] = None
    """For each token, the probabilities of each value in distribution_values."""

    uncalibrated_simulation: Optional["SequenceSimulation"] = None
    """The result of the simulation before calibration."""


@dataclass
class ScoredSequenceSimulation(Serializable):
    """
    SequenceSimulation result with a score (for that sequence only) and ground truth
    activations.
    """

    distance: int
    """Distance of the sequence from the original sequence."""

    simulation: SequenceSimulation
    """The simulation result."""

    true_activations: list[int]
    """The actual neuron activations for comparison."""

    ev_correlation_score: float | str
    """Correlation between expected values and ground truth."""

    rsquared_score: int
    """R-squared score (always 0 for compatibility)."""

    absolute_dev_explained_score: int
    """Absolute deviation explained score (always 0 for compatibility)."""


@dataclass
class ScoredSimulation(Serializable):
    """Result of scoring a neuron simulation on multiple sequences."""

    distance: int
    """Distance of the sequence from the original sequence."""

    scored_sequence_simulations: list[ScoredSequenceSimulation]
    """Individual sequence results."""

    ev_correlation_score: float | str
    """Correlation score, or "nan" if cannot be computed."""

    rsquared_score: int
    """R-squared score (always 0 for compatibility)."""

    absolute_dev_explained_score: int
    """Absolute deviation explained score (always 0 for compatibility)."""


# === MODERN CLEAN TYPES ===


@dataclass
class SimulationResult:
    """Per-sequence simulation outcome.

    Attributes:
        tokens: Token strings for the simulated sequence (aligned with predictions).
        predicted_activations: Predicted activation values per token.
        true_activations: Ground-truth activation values per token.
        correlation: Pearson correlation between true and predicted activations
            for this sequence.
        quantile: Quantile identifier associated with the sequence (e.g., for
            grouping results).
    """

    tokens: List[str]
    predicted_activations: List[float]
    true_activations: List[int]
    correlation: float
    quantile: int


@dataclass
class AggregateResult:
    """Aggregated results for a group of sequences.

    Attributes:
        quantile: Group identifier. Use -1 to represent the overall/combined
            group when both activating and non-activating sequences are present.
        correlation: Pearson correlation computed over all tokens from all
            sequences in the group (concatenated), not an average of per-sequence
            correlations.
        sequence_count: Number of sequences included in the group.
        sequences: The per-sequence results that were aggregated.
    """

    quantile: int  # -1 for "overall", 0+ for actual quantiles
    correlation: float
    sequence_count: int
    sequences: List[SimulationResult]


# === CONVERSION FUNCTIONS ===


def convert_to_legacy_format(results: List[AggregateResult]) -> List[ScoredSimulation]:
    """Convert aggregates to the standard scoring format.

    Transforms ``AggregateResult`` objects into ``ScoredSimulation`` structures
    used by the public scoring API. Distances are mapped as follows:
    - Overall group (quantile == -1) → distance 0
    - Quantile group q ≥ 0 → distance q + 1

    Sequence-level fields are preserved and represented as
    ``ScoredSequenceSimulation`` with expected activations stored in a
    ``SequenceSimulation``. Correlation values that are NaN are serialized as the
    string "nan" for downstream JSON compatibility.

    Args:
        results: Aggregated results to convert.

    Returns:
        List of ``ScoredSimulation`` instances suitable for serialization and
        external consumption.
    """
    legacy_results = []

    for aggregate in results:
        # Convert each sequence to legacy format
        legacy_sequences = []
        for seq in aggregate.sequences:
            sequence_simulation = SequenceSimulation(
                tokens=seq.tokens,
                expected_activations=seq.predicted_activations,
                activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
                distribution_values=[],  # Empty - not used in simplified version
                distribution_probabilities=[],  # Empty - not used in simplified version
                uncalibrated_simulation=None,
            )

            # Convert correlation to serializable format
            correlation_score = seq.correlation
            if np.isnan(correlation_score):
                correlation_score = "nan"
            else:
                correlation_score = float(correlation_score)

            scored_sequence = ScoredSequenceSimulation(
                distance=seq.quantile,
                simulation=sequence_simulation,
                true_activations=seq.true_activations,
                ev_correlation_score=correlation_score,
                rsquared_score=0,
                absolute_dev_explained_score=0,
            )
            legacy_sequences.append(scored_sequence)

        aggregate_correlation = aggregate.correlation
        if np.isnan(aggregate_correlation):
            aggregate_correlation = "nan"
        else:
            aggregate_correlation = float(aggregate_correlation)

        # Create legacy ScoredSimulation with distance mapping:
        #  - overall (-1) → 0
        #  - quantile q ≥ 0 → q + 1
        legacy_distance = 0 if aggregate.quantile == -1 else aggregate.quantile + 1

        legacy_result = ScoredSimulation(
            distance=legacy_distance,
            scored_sequence_simulations=legacy_sequences,
            ev_correlation_score=aggregate_correlation,
            rsquared_score=0,
            absolute_dev_explained_score=0,
        )
        legacy_results.append(legacy_result)

    return legacy_results


def _fix_nan(val: float) -> float:
    """Return 0.0 when ``val`` is NaN; otherwise return ``val`` as a float."""
    return 0.0 if np.isnan(val) else float(val)
