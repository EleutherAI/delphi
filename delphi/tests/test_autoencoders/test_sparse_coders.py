import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from delphi.config import RunConfig

# Import the function to be tested
from delphi.sparse_coders import load_hooks_sparse_coders


# A simple dummy run configuration for testing.
class DummyRunConfig:
    def __init__(self, sparse_model, hookpoints):
        self.sparse_model = sparse_model
        self.hookpoints = hookpoints
        # Additional required fields can be added here if needed.
        self.model = "dummy_model"
        self.hf_token = ""

    @property
    def __class__(self) -> type:  # type: ignore
        return RunConfig


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(x)


# A minimal dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        # For the sparsify branch, assume the model has an attribute named as the hook.
        self.layers = nn.ModuleList([DummyLayer() for _ in range(4)])
        # While we force the gemmascope hookpoint in the loader
        self.device = "cpu"

    def forward(self, x):
        return x

    @property
    def __class__(self) -> type:  # type: ignore
        return PreTrainedModel


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def run_cfg_sparsify():
    # The 'sparse_model' string should NOT contain "gemma" in order
    # to trigger the sparsify branch.
    return DummyRunConfig(
        sparse_model="EleutherAI/sae-pythia-70m-32k",
        hookpoints=["layers.4.mlp", "layers.0"],
    )


@pytest.fixture
def run_cfg_gemma():
    # The 'sparse_model' string here contains "gemma" to trigger the gemma branch.
    # The hookpoint must be in the format "layers_{layer}/width_{size}/average_l0_{l0}"
    return DummyRunConfig(
        sparse_model="google/gemma-scope-2b-pt-res/",
        hookpoints=[
            "layer_12/width_131k/average_l0_67",
            "layer_12/width_16k/average_l0_22",
        ],
    )


def test_retrieve_autoencoders_from_sparsify(dummy_model, run_cfg_sparsify):
    """
    Tests that load_hooks_sparse_coders retrieves autoencoders from Sparsify.
    """
    hookpoint_to_sparse_encode, _ = load_hooks_sparse_coders(
        dummy_model, run_cfg_sparsify
    )
    # Verify that we received a dictionary of autoencoders.
    assert (
        isinstance(hookpoint_to_sparse_encode, dict)
        and len(hookpoint_to_sparse_encode) > 0
    ), "No autoencoders retrieved from the Sparsify branch."

    # Validate that at least one autoencoder is callable.
    for key, autoencoder in hookpoint_to_sparse_encode.items():
        dummy_input = torch.randn(2, 512)
        try:
            _ = autoencoder(dummy_input)
        except Exception as e:
            pytest.fail(
                f"Autoencoder '{key}' from the Sparsify branch failed when called: {e}"
            )
        # Optionally, further tests can check output shapes or values.


def test_retrieve_autoencoders_from_gemma(dummy_model, run_cfg_gemma):
    """
    Tests that load_hooks_sparse_coders retrieves autoencoders from Gemma.
    """
    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        dummy_model, run_cfg_gemma
    )
    # Verify that we received a dictionary of autoencoders.
    assert (
        isinstance(hookpoint_to_sparse_encode, dict)
        and len(hookpoint_to_sparse_encode) > 0
    ), "No autoencoders retrieved from the Gemma branch."

    # Validate that at least one autoencoder is callable.
    for key, autoencoder in hookpoint_to_sparse_encode.items():
        dummy_input = torch.randn(2, 2304)
        try:
            _ = autoencoder(dummy_input)
        except Exception as e:
            pytest.fail(
                f"Autoencoder '{key}' from the Gemma branch failed when called: {e}"
            )
