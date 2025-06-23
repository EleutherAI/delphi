from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..activations.activations import ActivationRecord
from ..explanations.explanations import (
    ScoredSequenceSimulation,
    ScoredSimulation,
    SequenceSimulation,
)
from ..explanations.simulator import NeuronSimulator


def flatten_list(list_of_lists: Sequence[Sequence[Any]]) -> list[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def correlation_score(
    real_activations: Sequence[float] | np.ndarray,
    predicted_activations: Sequence[float] | np.ndarray,
) -> float:
    return np.corrcoef(real_activations, predicted_activations)[0, 1]


def score_from_simulation(
    real_activations: ActivationRecord,
    simulation: SequenceSimulation,
    score_function: Callable[
        [Sequence[float] | np.ndarray, Sequence[float] | np.ndarray], float
    ],
) -> float:
    if len(simulation.expected_activations) > 0:
        return score_function(
            real_activations.activations, simulation.expected_activations
        )
    else:
        return 0


def rsquared_score_from_sequences(
    real_activations: Sequence[float] | np.ndarray,
    predicted_activations: Sequence[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(
            np.square(np.array(real_activations) - np.array(predicted_activations))
        )
        / np.mean(np.square(np.array(real_activations)))
    )


def absolute_dev_explained_score_from_sequences(
    real_activations: Sequence[float] | np.ndarray,
    predicted_activations: Sequence[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(np.abs(np.array(real_activations) - np.array(predicted_activations)))
        / np.mean(np.abs(np.array(real_activations)))
    )


async def _simulate_and_score_sequence(
    simulator: NeuronSimulator, activations: ActivationRecord, quantile: int
) -> ScoredSequenceSimulation:
    """Score an explanation of a neuron by how well it predicts activations
    on a sentence."""

    simulation = await simulator.simulate(activations.tokens)
    logging.debug(simulation)
    rsquared_score = 0
    absolute_dev_explained_score = 0
    scored_sequence_simulation = ScoredSequenceSimulation(
        distance=quantile,
        simulation=simulation,
        true_activations=activations.activations,
        ev_correlation_score=score_from_simulation(
            activations, simulation, correlation_score
        ),
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )
    return scored_sequence_simulation


def fix_nan(val):
    if np.isnan(val):
        return "nan"
    else:
        return float(val)


def default(scored_simulation: ScoredSequenceSimulation) -> ScoredSequenceSimulation:
    ev_correlation_score = scored_simulation.ev_correlation_score

    ev_correlation_score = fix_nan(ev_correlation_score)

    return ScoredSequenceSimulation(
        distance=scored_simulation.distance,
        simulation=scored_simulation.simulation,
        true_activations=scored_simulation.true_activations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=scored_simulation.rsquared_score,
    )


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: list[ScoredSequenceSimulation],
    distance: int,
) -> ScoredSimulation:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is
    non-trivial for EV scores, since we want to calculate the correlation over all
    activations from all sequences at once rather than simply averaging
    per-sequence correlations.
    """
    all_true_activations: list[float] = []
    all_expected_values: list[float] = []
    for scored_sequence_simulation in scored_sequence_simulations:
        all_true_activations.extend(scored_sequence_simulation.true_activations or [])
        all_expected_values.extend(
            scored_sequence_simulation.simulation.expected_activations
        )
    ev_correlation_score = (
        correlation_score(all_true_activations, all_expected_values)
        if (len(all_true_activations) > 0 and len(all_expected_values) > 0)
        else 0
    )
    rsquared_score = 0
    absolute_dev_explained_score = 0

    scored_sequence_simulations = [default(s) for s in scored_sequence_simulations]

    ev_correlation_score = fix_nan(ev_correlation_score)

    return ScoredSimulation(
        distance=distance,
        scored_sequence_simulations=scored_sequence_simulations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=float(rsquared_score),
        absolute_dev_explained_score=float(absolute_dev_explained_score),
    )


async def simulate_and_score(
    simulator: NeuronSimulator,
    activation_records: Sequence[ActivationRecord],
    non_activation_records: Sequence[ActivationRecord],
) -> list[ScoredSimulation]:
    """
    Score an explanation of a neuron by how well it predicts activations
    on the given text sequences.
    """
    scored_sequence_simulations = await asyncio.gather(
        *[
            _simulate_and_score_sequence(
                simulator,
                activation_record,
                activation_record.quantile,
            )
            for activation_record in activation_records
        ]
    )

    if len(non_activation_records) > 0:
        non_activating_scored_seq_simulations = await asyncio.gather(
            *[
                _simulate_and_score_sequence(
                    simulator, non_activation_record, non_activation_record.quantile
                )
                for non_activation_record in non_activation_records
            ]
        )
    else:
        non_activating_scored_seq_simulations = []

    values: list[ScoredSimulation] = []
    simulations_per_distance = {}
    without_errors: list[ScoredSequenceSimulation] = []
    for sequence in scored_sequence_simulations:

        if len(sequence.simulation.expected_activations) > 0:
            if sequence.distance + 1 not in simulations_per_distance:
                simulations_per_distance[sequence.distance + 1] = []
            simulations_per_distance[sequence.distance + 1].append(sequence)
            without_errors.append(sequence)
    for quantile in simulations_per_distance.keys():
        values.append(
            aggregate_scored_sequence_simulations(
                simulations_per_distance[quantile], quantile
            )
        )

    if len(non_activation_records) > 0:
        for sequence in non_activating_scored_seq_simulations:
            if len(sequence.simulation.expected_activations) > 0:
                without_errors.append(sequence)
        values.append(aggregate_scored_sequence_simulations(without_errors, 0))
    return values
