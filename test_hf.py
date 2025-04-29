#%%
import random

import torch
from datasets import Dataset


def arbitrary_shape_dict():
    for _ in range(10):
        length = random.randint(1, 10)
        data_dict = {
            "locs": torch.randint(0, 100, (length, 3)),
            "vals": torch.rand(length),
        }
        for i in range(length):
            yield {k: v[i] for k, v in data_dict.items()}

Dataset.from_generator(arbitrary_shape_dict)
#%%
from safetensors.torch import load_file
st = load_file("results/sae-pkm/baseline/.model.layers.10.mlp/0_3685.safetensors")
#%%
len(st["activations"]) / st["tokens"].numel()