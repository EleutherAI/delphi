import asyncio
import logging
from typing import Callable

import numpy as np

from delphi.latents import ActivatingExample, NonActivatingExample

from ..activations.activation_records import ActivationRecord
from ..scoring.simulator import NeuronSimulator
from .dataclasses import ScoredSequenceSimulation, ScoredSimulation, SequenceSimulation


def correlation_score(
    real_activations: list[float] | np.ndarray,
    predicted_activations: list[float] | np.ndarray,
) -> float:
    return float(np.corrcoef(real_activations, predicted_activations)[0, 1])


def score_from_simulation(
    real_activations: ActivationRecord,
    simulation: SequenceSimulation,
    score_function: Callable[
        [list[float] | np.ndarray, list[float] | np.ndarray], float
    ],
) -> float:
    if len(simulation.expected_activations) > 0:
        return score_function(
            real_activations.activations, simulation.expected_activations
        )
    else:
        return 0


def rsquared_score_from_lists(
    real_activations: list[float] | np.ndarray,
    predicted_activations: list[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(
            np.square(np.array(real_activations) - np.array(predicted_activations))
        )
        / np.mean(np.square(np.array(real_activations)))
    )


def absolute_dev_explained_score_from_lists(
    real_activations: list[float] | np.ndarray,
    predicted_activations: list[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(np.abs(np.array(real_activations) - np.array(predicted_activations)))
        / np.mean(np.abs(np.array(real_activations)))
    )


async def _simulate_and_score_list(
    simulator: NeuronSimulator, example: ActivatingExample | NonActivatingExample
) -> ScoredSequenceSimulation | None:
    """Score an explanation of a neuron by how well it predicts activations
    on a sentence."""

    simulation = await simulator.simulate(example.str_tokens)
    # if simulation.expected_activations is empty, return None
    if len(simulation.expected_activations) == 0:
        return None
    logging.debug(simulation)
    rsquared_score = 0
    absolute_dev_explained_score = 0

    match example:
        case ActivatingExample():
            distance = example.quantile
            activating = True
        case NonActivatingExample():
            distance = example.distance
            activating = False
    activation_record = ActivationRecord(
        example.str_tokens, example.activations.tolist()
    )
    scored_sequence_simulation = ScoredSequenceSimulation(
        distance=distance,
        simulation=simulation,
        true_activations=activation_record.activations,
        ev_correlation_score=(
            score_from_simulation(activation_record, simulation, correlation_score)
            if activating
            else 0
        ),  # can't do EV when truth is 0
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )
    return scored_sequence_simulation


def fix_nan(val):
    if np.isnan(val):
        return "nan"
    else:
        return float(val)


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: list[ScoredSequenceSimulation],
    distance: int,
) -> ScoredSimulation:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is
    non-trivial for EV scores, since we want to calculate the correlation over all
    activations from all lists at once rather than simply averaging
    per-list correlations.
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
    rsquared_score = (
        rsquared_score_from_lists(all_true_activations, all_expected_values)
        if (len(all_true_activations) > 0 and len(all_expected_values) > 0)
        else 0
    )
    absolute_dev_explained_score = (
        absolute_dev_explained_score_from_lists(
            all_true_activations, all_expected_values
        )
        if (len(all_true_activations) > 0 and len(all_expected_values) > 0)
        else 0
    )

    ev_correlation_score = fix_nan(ev_correlation_score)
    absolute_dev_explained_score = fix_nan(absolute_dev_explained_score)
    rsquared_score = fix_nan(rsquared_score)

    return ScoredSimulation(
        distance=distance,
        scored_sequence_simulations=scored_sequence_simulations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )


async def simulate_and_score(
    simulator: NeuronSimulator,
    activation_records: list[ActivatingExample],
    non_activation_records: list[NonActivatingExample],
) -> ScoredSimulation:
    """
    Score an explanation of a neuron by how well it predicts activations
    on the given text lists.
    """
    scored_sequence_simulations = await asyncio.gather(
        *[
            _simulate_and_score_list(simulator, activation_record)
            for activation_record in activation_records
        ]
    )
    if len(non_activation_records) > 0:
        non_activating_scored_seq_simulations = await asyncio.gather(
            *[
                _simulate_and_score_list(simulator, non_activation_record)
                for non_activation_record in non_activation_records
            ]
        )

    # validate predictions, remove those that are None
    scored_sequence_simulations = [
        sequence for sequence in scored_sequence_simulations if sequence is not None
    ]

    values = []
    scores_per_distance = {}
    for sequence in scored_sequence_simulations:
        distance = sequence.distance
        if distance not in scores_per_distance:
            scores_per_distance[distance] = []
        scores_per_distance[distance].append(sequence)

    for distance, sequences in scores_per_distance.items():
        values.append(aggregate_scored_sequence_simulations(sequences, distance + 1))

    # plus an aggregate score for all distances
    values.append(aggregate_scored_sequence_simulations(scored_sequence_simulations, 0))

    # plus an aggregate score for non-activated + all distances
    all_data = scored_sequence_simulations + non_activating_scored_seq_simulations
    values.append(aggregate_scored_sequence_simulations(all_data, -1))

    return values
