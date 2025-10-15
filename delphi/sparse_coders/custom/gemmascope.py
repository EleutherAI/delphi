from functools import partial

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


def load_gemma_autoencoders(
    model_path: str,
    ae_layers: list[int],
    average_l0s: list[int],
    sizes: list[str],
    type: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = torch.device("cuda"),
) -> dict[str, nn.Module]:
    saes = {}

    for layer, size, l0 in zip(ae_layers, sizes, average_l0s):
        path = f"layer_{layer}/width_{size}/average_l0_{l0}"
        sae = JumpReluSae.from_pretrained(model_path, path, device)

        sae.to(dtype)

        assert type in [
            "res",
            "mlp",
        ], "Only res and mlp are supported for gemma autoencoders"
        hookpoint = (
            f"layers.{layer}"
            if type == "res"
            else f"layers.{layer}.post_feedforward_layernorm"
        )

        saes[hookpoint] = sae

    return saes


def load_gemma_hooks(
    model_path: str,
    ae_layers: list[int],
    average_l0s: list[int],
    sizes: list[str],
    type: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = torch.device("cuda"),
):
    saes = load_gemma_autoencoders(
        model_path,
        ae_layers,
        average_l0s,
        sizes,
        type,
        dtype,
        device,
    )
    hookpoint_to_sparse_encode = {}
    for hookpoint, sae in saes.items():

        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded

        hookpoint_to_sparse_encode[hookpoint] = partial(_forward, sae)

    return hookpoint_to_sparse_encode


# This is from the GemmaScope tutorial
# https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp#scrollTo=WYfvS97fAFzq
class JumpReluSae(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    @classmethod
    def from_pretrained(cls, model_name_or_path, position, device):
        path_to_params = hf_hub_download(
            repo_id=model_name_or_path,
            filename=f"{position}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1])
        model.load_state_dict(pt_params)
        if device == "cuda" or (
            isinstance(device, torch.device) and device.type == "cuda"
        ):
            model.cuda()
        return model
