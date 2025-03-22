from functools import partial

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from pathlib import Path

"""
- model_path should be a path to the repository with the weights
- hookpoint should be the same name as for the transformer_lens hookpoint
"""
def load_hsae(
    model_path: str,
    hookpoint: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = torch.device("cuda"),
) -> dict[str, nn.Module]:
    saes = {}

    # model path should be where sae_weights.safetensors is located
    sae = HSae.from_pretrained(model_path, device)
    sae.to(dtype)

    # the hookpoints have names of the form blocks.8.hook_resid_post
    saes[hookpoint] = sae

    return saes

def load_hsae_hooks(
    model_path: str,
    hookpoint: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = torch.device("cuda"),
):
    saes = load_hsae(
        model_path,
        hookpoint,
        dtype,
        device,
    )
    hookpoint_to_sparse_encode = {}
    for hookpoint, sae in saes.items():

        def _forward(sae, x):
            encoded = sae.encode_concatenate(x)
            return encoded

        hookpoint_to_sparse_encode[hookpoint] = partial(_forward, sae)

    return hookpoint_to_sparse_encode

class HSae(nn.Module):
    def __init__(self, d_model, d_sae, d_sae_h):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.W_enc_h = nn.Parameter(torch.zeros(d_model, d_sae_h))
        self.W_dec_h = nn.Parameter(torch.zeros(d_sae_h, d_model))
        self.threshold_h = nn.Parameter(torch.zeros(d_sae_h))
        self.b_enc_h = nn.Parameter(torch.zeros(d_sae_h))
        self.b_dec_h = nn.Parameter(torch.zeros(d_model))

        self.A = self.encode_h(self.W_dec)

    def update_A(self, device="cuda"): 
        self.A = self.encode_h(self.W_dec).to(device)

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    
    def encode_double(self, input_acts):
        # acts has shape (batch_size, d_sae)
        acts = self.encode(input_acts)
        # A has shape (d_sae, d_sae_h)
        output = acts @ self.A
        return output
    
    def encode_concatenate(self, input_acts): 
        # acts has shape (batch_size, d_sae)
        acts = self.encode(input_acts)
        print(acts.shape)
        # A has shape (d_sae, d_sae_h)
        output = acts @ self.A
        print(self.A.shape)
        print(output.shape)
        return torch.cat((acts, output), dim=2)

    def encode_h(self, input_acts):
        pre_acts = (input_acts - self.b_dec_h) @ self.W_enc_h + self.b_enc_h
        mask = pre_acts > self.threshold_h
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    @classmethod
    def from_pretrained(cls, model_name_or_path, device):
        print(model_name_or_path / "sae_weights.safetensors")
        params = load_file(model_name_or_path / "sae_weights.safetensors")
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1], params["W_enc_h"].shape[1])
        model.load_state_dict(params)
        model.update_A(device)
        if device == "cuda":
            model.cuda()
        return model
