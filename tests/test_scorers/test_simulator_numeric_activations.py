from delphi.scorers.simulator.simulation.data_models import (
    ActivationRecord,
    calculate_max_activation,
    format_activation_records,
    normalize_activations,
    relu,
)
from delphi.scorers.simulator.simulation.simulator import NeuronSimulator


def test_relu_accepts_integer_sentinel_and_returns_float():
    assert relu(-1) == 0.0
    assert type(relu(2)) is float


def test_normalization_accepts_mixed_numeric_activations():
    assert normalize_activations([-1, 0, 2.5, 5], 5.0) == [0, 0, 5, 10]


def test_integer_records_can_be_normalized_and_formatted():
    record = ActivationRecord(tokens=["a", "b"], activations=[0, 2], quantile=0)
    maximum = calculate_max_activation([record])
    assert maximum == 2.0
    assert "b\t10" in format_activation_records([record], maximum)


def test_original_few_shot_prompt_builds_with_integer_sentinels():
    simulator = NeuronSimulator(object(), "cats", use_logprobs=True)
    prompt = simulator._make_logprob_prompt(["cat"])
    assert "cat\tunknown" in prompt[-1]["content"]
