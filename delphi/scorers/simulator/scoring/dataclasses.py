from dataclasses import dataclass
from typing import Optional

from simple_parsing import Serializable


@dataclass
class SequenceSimulation(Serializable):
    """The result of a simulation of neuron activations on one text sequence."""

    tokens: list[str]
    """The sequence of tokens that was simulated."""
    expected_activations: list[float]
    """Expected value of the possibly-normalized activation for
    each token in the sequence."""
    distribution_values: list[list[float]]
    """
    For each token in the sequence, a list of values from the discrete
    distribution of activations produced from simulation.
    Tokens will be included here if and only if they are in the top K=15
    tokens predicted by the simulator, and excluded otherwise.

    May be transformed to another unit by calibration. When we simulate a neuron,
    we produce a discrete distribution with values in the arbitrary discretized space
    of the neuron, e.g. 10% chance of 0, 70% chance of 1, 20% chance of 2.
    Which we store as distribution_values = [0, 1, 2],
    distribution_probabilities = [0.1, 0.7, 0.2]. When we transform the distribution to
    the real activation units, we can correspondingly transform the values of this
    distribution to get a distribution in the units of the neuron. e.g. if the mapping
    from the discretized space to the real activation unit of the neuron is f(x) = x/2,
    then the distribution becomes 10% chance of 0, 70% chance of 0.5, 20% chance of 1.
    Which we store as distribution_values = [0, 0.5, 1], distribution_probabilities =
    [0.1, 0.7, 0.2].
    """
    distribution_probabilities: list[list[float]]
    """
    For each token in the sequence, the probability of the corresponding value in
    distribution_values.
    """

    uncalibrated_simulation: Optional["SequenceSimulation"] = None
    """The result of the simulation before calibration."""


@dataclass
class ScoredSequenceSimulation(Serializable):
    """
    SequenceSimulation result with a score (for that sequence only) and ground truth
    activations.
    """

    distance: float | int
    """Quantile or neighbor distance"""
    simulation: SequenceSimulation
    """The result of a simulation of neuron activations."""
    true_activations: list[float]
    """Ground truth activations on the sequence (not normalized)"""
    ev_correlation_score: float
    """
    Correlation coefficient between the expected values of the normalized activations
    from the simulation and the unnormalized true activations of the neuron on the text
    sequence.
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real))
    """


@dataclass
class ScoredSimulation(Serializable):
    """Result of scoring a neuron simulation on multiple sequences."""

    distance: int
    """Distance of the sequence from the original sequence."""

    scored_sequence_simulations: list[ScoredSequenceSimulation]
    """ScoredSequenceSimulation for each sequence"""
    ev_correlation_score: Optional[float] = None
    """
    Correlation coefficient between the expected values of the normalized activations
    from the simulation and the unnormalized true activations on a dataset created from
    all score_results. (Note that this is not equivalent to averaging across sequences.)
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real)).
    """
