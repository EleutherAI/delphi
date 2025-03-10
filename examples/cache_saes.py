from nnsight import LanguageModel, NNsight
from simple_parsing import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from delphi.autoencoders.wrapper import AutoencoderLatents
from functools import partial
from delphi.config import CacheConfig
from delphi.latents import LatentCache
from delphi.utils import load_tokenized_data
from sparsify import Sae
from pathlib import Path
from torch import nn
from typing import Callable
from pprint import pprint
import json
import os

import torch._dynamo
torch._dynamo.config.suppress_errors = True


class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update(
            {
                prefix + "k": self.k,
                prefix + "postact_fn": self.postact_fn.__class__.__name__,
            }
        )
        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}


def main(cfg: CacheConfig, args):
    k = args.k
    expansion = args.expansion
    transcode = args.transcode
    pkm = args.pkm
    model_name = args.model_name
    sae_dir = Path(args.sae_dir)

    if args.path:
        sae_dir = Path(args.path)
        config_path = sae_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("No config.json found")
        config = json.loads(config_path.read_text())
        pprint(config)
        model_name = config["model"]
        k = config["sae"]["k"]
        expansion = config["sae"]["expansion_factor"]
        transcode = config["sae"]["transcode"]
    else:
        for sae_dir in sae_dir.glob("*"):
            config_path = sae_dir / "config.json"
            if not config_path.exists():
                continue
            config = json.loads(config_path.read_text())
            if config["sae"]["expansion_factor"] != expansion:
                continue
            if config["sae"]["encoder_pkm"] != pkm:
                continue
            if config["sae"]["encoder_kron"] != args.kron:
                continue
            if config["transcode"] != transcode:
                continue
            if config["sae"]["k"] != k:
                continue
            if config["sae"]["monet"] != args.monet:
                continue
            if model_name not in config["model"]:
                continue
            model_name = config["model"]
            break
        else:
            raise FileNotFoundError("No matching SAE found")
    print("Using config", sae_dir.name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = NNsight(model)
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    submodule_dict = {}

    name = sae_dir.name
    cache_save_dir = f"results/sae-pkm/{name}"
    os.makedirs(cache_save_dir, exist_ok=True)
    cache_save_dir = Path(cache_save_dir)

    device = "cuda:0"
    for hookpoint in config["hookpoints"]:
        weight_dir = sae_dir / hookpoint

        try:
            sae = Sae.load_from_disk(weight_dir, device=device).to(dtype=model.dtype)
        except RuntimeError:
            open("failed_to_load.txt", "a").write(str(weight_dir) + "\n")
            raise
        if not hookpoint.startswith(model.base_model_prefix):
            hookpoint = f"{model.base_model_prefix}.{hookpoint}"

        if (cache_save_dir / ("." + hookpoint)).exists():
            print("Skipping", hookpoint)
            continue
        else:
            print("Caching", (cache_save_dir / ("." + hookpoint)))

        def _forward(sae, k,x):
            encoded = sae.pre_acts(x)
            if k is not None:
                trained_k = k
            else:
                trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)

        atoms = hookpoint.split(".")
        submodule = model
        for atom in atoms:
            submodule = getattr(submodule, atom)
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae, k), width=sae.encoder.weight.shape[0],
            hookpoint=hookpoint
        )

        submodule_dict["." + hookpoint] = submodule.ae

    if not submodule_dict:
        print("All submodules have been cached")
        return
    else:
        print("Caching into", cache_save_dir)
        print("Submodules:", submodule_dict.keys())

    with model.edit("") as edited:
        for path, submodule in submodule_dict.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    tokens = load_tokenized_data(
        cfg.ctx_len,
        edited.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
    )
    cache = LatentCache(
        edited,
        submodule_dict,
        batch_size=cfg.batch_size,
    )
    # name=""
    # name += model_name.replace("/", "-")
    # name += f"_x{expansion}"
    # name += f"_k-{k}"
    # name += "_trans" if transcode else ""
    # name += "_pkm" if pkm else ""
    cache.run(10_000_000, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir=cache_save_dir
    )
    cache.save_config(
        save_dir=cache_save_dir,
        cfg=cfg,
        model_name=model_name,
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--sae_dir", type=str, default="../halutsae/sae-pkm/smollm")
    parser.add_argument("--path", type=str)
    parser.add_argument("--model_name", type=str,
                        # default="pythia-160m"
                        default="SmolLM"
                        )
    parser.add_argument("--expansion", type=int, default=32)
    parser.add_argument("--transcode", action="store_true")
    parser.add_argument("--pkm", action="store_true")
    parser.add_argument("--kron", action="store_true")
    parser.add_argument("--monet", action="store_true")
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()
    cfg = args.options
    print(cfg)

    main(cfg, args)
