from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch._dynamo
from simple_parsing import Serializable

torch._dynamo.config.suppress_errors = True


@dataclass
class AutoencoderConfig(Serializable):
    model_name_or_path: str = "model"
    autoencoder_type: Literal["SAE", "SAE_LENS", "NEURONS", "CUSTOM"] = "SAE"
    device: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class AutoencoderLatents(torch.nn.Module):
    """
    Unified wrapper for different types of autoencoders, compatible with nnsight.
    """

    def __init__(
        self,
        autoencoder: Any,
        forward_function: Callable,
        width: int,
        hookpoint: str,
    ) -> None:
        super().__init__()
        self.ae = autoencoder
        self._forward = forward_function
        self.width = width
        self.hookpoint = hookpoint

    def forward(self, x: torch.Tensor, hook=False) -> torch.Tensor:
        return self._forward(x)

    @classmethod
    def from_pretrained(cls, config: AutoencoderConfig, hookpoint: str, **kwargs):
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder_type = config.autoencoder_type
        model_name_or_path = config.model_name_or_path
        if autoencoder_type == "SAE":
            from sae import Sae

            local = kwargs.get("local", None)
            assert local is not None, "local must be specified for SAE"
            if local:
                sae = Sae.load_from_disk(
                    model_name_or_path + "/" + hookpoint, device=device, **kwargs
                )
            else:
                sae = Sae.load_from_hub(
                    model_name_or_path, hookpoint, device=device, **kwargs
                )
            forward_function = choose_forward_function(config, sae)
            width = sae.encoder.weight.shape[0]

        elif autoencoder_type == "SAE_LENS":
            from sae_lens import SAE

            sae, cfg_dict, sparsity = SAE.from_pretrained(
                # see other options in sae_lens/pretrained_saes.yaml
                release=model_name_or_path,
                sae_id=hookpoint,
                device=device,
            )
            forward_function = choose_forward_function(config, sae)
            width = sae.d_sae

        elif autoencoder_type == "NEURONS":
            raise NotImplementedError("Neurons autoencoder not implemented yet")

        elif autoencoder_type == "CUSTOM":
            # to use a custom autoencoder, you must make own
            # custom autoencoder class and implement the forward function.
            # it should have a specific name that you use here
            custom_name = config.kwargs.get("custom_name", None)
            if custom_name is None:
                raise ValueError("custom_name must be specified for CUSTOM autoencoder")
            if custom_name == "gemmascope":
                from .Custom.gemmascope import JumpReLUSAE

                sae = JumpReLUSAE.from_pretrained(model_name_or_path, hookpoint, device)
                forward_function = choose_forward_function(config, sae)
                width = sae.W_enc.data.shape[1]
            if custom_name == "openai":
                raise NotImplementedError("OpenAI autoencoder not implemented yet")
                # from Custom.openai import Autoencoder

                # path = f"{model_name_or_path}/{hookpoint}.pt"
                # state_dict = torch.load(path)
                # ae = Autoencoder.from_state_dict(state_dict=state_dict)

        else:
            raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")

        return cls(sae, forward_function, width, hookpoint)

    @classmethod
    def random(cls, config: AutoencoderConfig, hookpoint: str, **kwargs):
        pass


def choose_forward_function(autoencoder_config: AutoencoderConfig, autoencoder: Any):
    if autoencoder_config.autoencoder_type == "SAE":
        from .Custom.openai import ACTIVATIONS_CLASSES, TopK

        def _forward(sae, k, x):
            encoded = sae.pre_acts(x)
            if k is not None:
                trained_k = k
            else:
                trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)

        k = autoencoder_config.kwargs.get("k", None)
        return partial(_forward, autoencoder, k)

    elif autoencoder_config.autoencoder_type == "SAE_LENS":
        return autoencoder.encode

    elif autoencoder_config.autoencoder_type == "CUSTOM":
        if autoencoder_config.kwargs.get("custom_name", None) == "gemmascope":
            return autoencoder.encode
        else:
            raise ValueError(
                "Unsupported custom autoencoder: "
                f"{autoencoder_config.kwargs.get('custom_name', None)}"
            )


def get_submodule(
    model: Any, autoencoder_config: AutoencoderConfig, hookpoint: str
) -> Any:

    if autoencoder_config.autoencoder_type == "SAE":
        if "res" in hookpoint:
            submodule = model.model.get_submodule(hookpoint)
        elif "mlp" in hookpoint:
            layer = int(hookpoint.split(".")[-1])
            submodule = model.model.layers[layer].mlp
        else:
            raise ValueError(f"Unsupported hookpoint: {hookpoint}")
        return submodule
    elif autoencoder_config.autoencoder_type == "SAE_LENS":
        raise NotImplementedError("SAE_LENS not implemented yet")
        # return model.get_submodule(hookpoint)
    elif autoencoder_config.autoencoder_type == "CUSTOM":
        if autoencoder_config.kwargs.get("custom_name", None) == "gemmascope":
            layer = int(hookpoint.split("/")[0].split("_")[-1])
            model_name = autoencoder_config.model_name_or_path
            if "res" in model_name:
                submodule = model.model.layers[layer]
            if "mlp" in model_name:
                submodule = model.model.layers[layer].post_feedforward_layernorm
            return submodule


def hook_submodule(
    submodule: Any, model: Any, module_path: str, autoencoder_config: AutoencoderConfig
) -> Tuple[Any, Any]:
    # TODO: This should take into account the autoencoder config,
    # but for now I think this is valid for all
    with model.edit("") as edited:
        if "embed" not in module_path and "mlp" not in module_path:
            acts = submodule.output[0]
        else:
            acts = submodule.output
        submodule.ae(acts, hook=True)
    return submodule, edited


def load_autoencoder_into_model(
    model: Any, autoencoder_config: AutoencoderConfig, hookpoints: List[str], **kwargs
) -> Tuple[Dict[str, Any], Any]:
    """
    Load an autoencoder and hook it into the model using nnsight.

    Args:
        model (Any): The main model to hook the autoencoder into.
        autoencoder_config (AutoencoderConfig): Configuration for the autoencoder.

    Returns:
        Tuple[List[Any], Any]: The list of submodules with the autoencoder attached
        Model with the autoencoder hooked in
    """

    submodules = {}
    edited_model = model
    assert hookpoints is not None, "Hookpoints must be specified in autoencoder_config"
    for module_path in hookpoints:
        autoencoder = AutoencoderLatents.from_pretrained(
            autoencoder_config,
            module_path,
        )
        submodule = get_submodule(edited_model, autoencoder_config, module_path)
        submodule.ae = autoencoder
        submodule, edited_model = hook_submodule(
            submodule, edited_model, module_path, autoencoder_config
        )

        submodules[submodule.path] = submodule

    return submodules, edited_model
