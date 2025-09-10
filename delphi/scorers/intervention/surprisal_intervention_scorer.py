import functools
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ...latents import LatentRecord
from ..scorer import Scorer, ScorerResult


@dataclass
class SurprisalInterventionResult:
    """
    Detailed results from the SurprisalInterventionScorer.

    Attributes:
        score: The final computed score.
        avg_kl: The average KL divergence between clean & intervened
                next-token distributions.
        explanation: The explanation string that was scored.
    """

    score: float
    avg_kl: float
    explanation: str
    tuned_strength: float


class SurprisalInterventionScorer(Scorer):
    """
    Implements the Surprisal / Log-Probability Intervention Scorer.

    This scorer evaluates an explanation for a model's latent feature by measuring
    how much an intervention in the feature's direction increases the model's belief
    (log-probability) in the explanation. The change in log-probability is normalized
    by the intervention's strength, measured by the KL divergence between the clean
    and intervened next-token distributions.

    Reference: Paulo et al., "Automatically Interpreting Millions of Features in LLMs"
    (https://arxiv.org/pdf/2410.13928), Section 3.3.5[cite: 206, 207].

    Pipeline:
      1. For a small set of activating prompts:
         a. Generate a continuation and get the next-token distribution ("clean").
         b. Add directional vector for the feature to the activations ("intervened").
      2. Compute the log-probability of the explanation conditioned on both the clean
         and intervened generated texts: log P(explanation | text).
      3. Compute KL divergence between the clean & intervened next-token distributions.
      4. The final score is the mean change in explanation log-prob, divided by the
         mean KL divergence:
         score = mean(log_prob_intervened - log_prob_clean) / (mean_KL + ε).
    """

    name = "surprisal_intervention"

    def __init__(self, subject_model: Any, explainer_model: Any = None, **kwargs):
        """
        Args:
            subject_model: The language model to generate from and score with.
            explainer_model: A model (e.g., an SAE) used to get feature directions.
            **kwargs: Configuration options.
                strength (float): The magnitude of the intervention. Default: 5.0.
                num_prompts (int): Number of activating examples to test. Default: 3.
                max_new_tokens (int): Max tokens to generate for continuations.
                hookpoint (str): The module name (e.g., 'transformer.h.10.mlp')
                                 for the intervention.
        """
        self.subject_model = subject_model
        self.explainer_model = explainer_model
        self.strength = float(kwargs.get("strength", 5.0))
        self.num_prompts = int(kwargs.get("num_prompts", 3))
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 20))
        self.hookpoints = kwargs.get("hookpoints")

        self.target_kl = float(kwargs.get("target_kl", 1.0))
        self.kl_tolerance = float(kwargs.get("kl_tolerance", 0.1))
        self.max_search_steps = int(kwargs.get("max_search_steps", 15))

        if len(self.hookpoints):
            self.hookpoint_str = self.hookpoints[0]

        if hasattr(subject_model, "tokenizer"):
            self.tokenizer = subject_model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.subject_model.config.pad_token_id = self.tokenizer.eos_token_id


    def _get_device(self) -> torch.device:
        """Safely gets the device of the subject model."""
        try:
            return next(self.subject_model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _find_layer(self, model: Any, name: str) -> torch.nn.Module:
        """Resolves a module by its dotted path name."""
        if name is None:
            raise ValueError("Hookpoint name is not configured.")
        current = model
        for part in name.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current


    def _get_full_hookpoint_path(self, hookpoint_str: str) -> str:
            """
            Heuristically finds the model's prefix and constructs the full hookpoint path string.
            e.g., 'layers.6.mlp' -> 'model.layers.6.mlp'
            """
            # Heuristically find the model prefix.
            prefix = None
            for p in ["gpt_neox", "transformer", "model"]:
                if hasattr(self.subject_model, p):
                    candidate_body = getattr(self.subject_model, p)
                    if hasattr(candidate_body, "h") or hasattr(candidate_body, "layers"):
                        prefix = p
                        break
            
            return f"{prefix}.{hookpoint_str}" if prefix else hookpoint_str


    def _resolve_hookpoint(self, model: Any, hookpoint_str: str) -> Any:
        """
        Finds and returns the actual module object for a given hookpoint string.
        """
        full_path = self._get_full_hookpoint_path(hookpoint_str)
        try:
            return self._find_layer(model, full_path)
        except AttributeError as e:
            raise AttributeError(f"""Could not resolve path '{full_path}'. 
                                     Model structure might be unexpected. 
                                     Original error: {e}""")


    def _sanitize_examples(self, examples: List[Any]) -> List[Dict[str, Any]]:
        """
        Function used for formatting results to run smoothly in the delphi pipeline
        """
        sanitized = []
        for ex in examples:
            if hasattr(ex, "str_tokens") and ex.str_tokens is not None:
                sanitized.append({"str_tokens": ex.str_tokens})

            elif isinstance(ex, dict) and "str_tokens" in ex:
                sanitized.append(ex)

            elif isinstance(ex, str):
                sanitized.append({"str_tokens": [ex]})

            elif isinstance(ex, (list, tuple)):
                sanitized.append({"str_tokens": [str(t) for t in ex]})

            else:
                sanitized.append({"str_tokens": [str(ex)]})

        return sanitized


    async def __call__(self, record: LatentRecord) -> ScorerResult:

        record_copy = copy.deepcopy(record)

        raw_examples = getattr(record_copy, "test", []) or []

        if not raw_examples:
            result = SurprisalInterventionResult(
                score=0.0, avg_kl=0.0, explanation=record_copy.explanation
            )
            return ScorerResult(record=record, score=[result.__dict__])

        examples = self._sanitize_examples(raw_examples)

        prompts = ["".join(ex["str_tokens"]) for ex in examples[: self.num_prompts]]

        #Step 1 - Truncate prompts before tuning or scoring.
        truncated_prompts = [
            await self._truncate_prompt(p, record_copy) for p in prompts
        ]

        #Step 2 - Tune intervention strength to match target KL.
        tuned_strength, initial_kl = await self._tune_strength(truncated_prompts, record_copy)
        
        total_diff = 0.0
        total_kl = 0.0
        n = 0

        for prompt in truncated_prompts:
            clean_text, clean_logp_dist = await self._generate_with_intervention(
                prompt, record_copy, strength=0.0, get_logp_dist=True
            )
            int_text, int_logp_dist = await self._generate_with_intervention(
                prompt, record_copy, strength=tuned_strength, get_logp_dist=True
            )
            
            logp_clean = await self._score_explanation(clean_text, record_copy.explanation)
            logp_int = await self._score_explanation(int_text, record_copy.explanation)
            
            p_clean = torch.exp(clean_logp_dist)
            kl_div = F.kl_div(int_logp_dist, p_clean, reduction="sum", log_target=False).item()

            total_diff += logp_int - logp_clean
            total_kl += kl_div
            n += 1

        avg_diff = total_diff / n if n > 0 else 0.0
        avg_kl = total_kl / n if n > 0 else 0.0
        
        #Final score is the average difference, not normalized by KL.
        final_score = avg_diff

        final_output_list = []
        for i, ex in enumerate(examples[: self.num_prompts]):
            final_output_list.append(
                {
                    "str_tokens": ex["str_tokens"],
                    "truncated_prompt": truncated_prompts[i],
                    "final_score": final_score,
                    "avg_kl_divergence": avg_kl,
                    "tuned_strength": tuned_strength,
                    "target_kl": self.target_kl,
                    # Placeholder keys
                    "distance": None, "activating": None, "prediction": None,
                    "correct": None, "probability": None, "activations": None,
                }
            )
        return ScorerResult(record=record_copy, score=final_output_list)


    async def _get_latent_activations(self, prompt: str, record: LatentRecord) -> torch.Tensor:
        """
        Runs a forward pass to get the SAE's latent activations for a prompt.
        """
        device = self._get_device()
        hookpoint_str = self.hookpoint_str or getattr(record, "hookpoint", None)
        sae = self._get_sae_for_hookpoint(hookpoint_str, record)
        if not sae:
            return torch.empty(0) # Return empty tensor if no SAE to encode with

        captured_hidden_states = []
        def capture_hook(module, inp, out):
            hidden_states = out[0] if isinstance(out, tuple) else out
            captured_hidden_states.append(hidden_states.detach().cpu())

        layer_to_hook = self._resolve_hookpoint(self.subject_model, hookpoint_str)
        handle = layer_to_hook.register_forward_hook(capture_hook)
        
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                self.subject_model(input_ids)
        finally:
            handle.remove()

        if not captured_hidden_states:
            return torch.empty(0)

        hidden_states = captured_hidden_states[0].to(device)

        encoding_result = sae.encode(hidden_states)
        feature_acts = encoding_result[2]

        return feature_acts[0, :, record.feature_id].cpu()


    async def _truncate_prompt(self, prompt: str, record: LatentRecord) -> str:
        """
        Truncates a prompt to end just before the first token where the latent activates.
        """
        activations = await self._get_latent_activations(prompt, record)
        if activations.numel() == 0:
            return prompt # Cannot truncate if no activations found

        # Find the index of the first token with non-zero activation
        first_activation_idx = (activations > 1e-6).nonzero(as_tuple=True)[0]
        
        if first_activation_idx.numel() > 0:
            truncation_point = first_activation_idx[0].item()
            if truncation_point > 0:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
                truncated_ids = input_ids[:truncation_point]
                return self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        
        return prompt


    async def _tune_strength(
        self, prompts: List[str], record: LatentRecord
    ) -> Tuple[float, float]:
        """
        Performs a binary search to find the intervention strength that matches `target_kl`.
        """
        low_strength, high_strength = 0.0, 40.0 # Heuristic search range
        best_strength = self.target_kl # Default to target_kl if search fails
        
        for _ in range(self.max_search_steps):
            mid_strength = (low_strength + high_strength) / 2
            
            # Estimate KL at mid_strength
            total_kl = 0.0
            n = 0
            for prompt in prompts:
                _, clean_logp = await self._generate_with_intervention(prompt, record, 0.0, True)
                _, int_logp = await self._generate_with_intervention(prompt, record, mid_strength, True)

                p_clean = torch.exp(clean_logp)
                kl_div = F.kl_div(int_logp, p_clean, reduction="sum", log_target=False).item()
                total_kl += kl_div
                n += 1
            
            current_kl = total_kl / n if n > 0 else 0.0
            
            if abs(current_kl - self.target_kl) < self.kl_tolerance:
                return mid_strength, current_kl

            if current_kl < self.target_kl:
                low_strength = mid_strength
            else:
                high_strength = mid_strength
            
            best_strength = mid_strength

        # Return the best found strength and the corresponding KL
        final_kl = await self._calculate_avg_kl(prompts, record, best_strength)
        return best_strength, final_kl


    async def _calculate_avg_kl(self, prompts: List[str], record: LatentRecord, strength: float) -> float:
        total_kl = 0.0
        n = 0
        for prompt in prompts:
            _, clean_logp = await self._generate_with_intervention(prompt, record, 0.0, True)
            _, int_logp = await self._generate_with_intervention(prompt, record, strength, True)
            p_clean = torch.exp(clean_logp)
            kl_div = F.kl_div(int_logp, p_clean, reduction="sum", log_target=False).item()
            total_kl += kl_div
            n += 1
        return total_kl / n if n > 0 else 0.0


    async def _generate_with_intervention(
        self, prompt: str, record: LatentRecord, strength: float, get_logp_dist: bool = False
    ) -> Tuple[str, torch.Tensor]:
        device = self._get_device()
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        input_ids = enc["input_ids"].to(device)

        hooks = []
        if strength > 0:
            hookpoint_str = self.hookpoint_str or getattr(record, "hookpoint", None)
            if hookpoint_str is None:
                raise ValueError("No hookpoint string specified for intervention.")
                
            layer_to_hook = self._resolve_hookpoint(self.subject_model, hookpoint_str)
            sae = self._get_sae_for_hookpoint(hookpoint_str, record)
            if not sae:
                raise ValueError(f"Could not find a valid SAE for hookpoint {hookpoint_str}")


            def hook_fn(module, inp, out):
                hidden_states = out[0] if isinstance(out, tuple) else out
                original_dtype = hidden_states.dtype
                
                # Get the latent dimension from the SAE's encoder
                d_latent = sae.encoder.out_features
                sae_device = sae.encoder.weight.device

                # --- Compute the decoder vector for the target feature ---
                # 1. Create a one-hot activation for our single feature.
                one_hot_activation = torch.zeros(1, 1, d_latent, device=sae_device)
                one_hot_activation[0, 0, record.feature_id] = 1.0

                # 2. Create the corresponding indices needed for the decode method.
                indices = torch.tensor([[[record.feature_id]]], device=sae_device, dtype=torch.long)

                # 3. Decode this one-hot vector to get the feature's direction in the hidden space.
                # We subtract the decoded zero vector to remove any decoder bias.
                decoded_zero = sae.decode(torch.zeros_like(one_hot_activation), indices)
                decoder_vector = sae.decode(one_hot_activation, indices) - decoded_zero
                decoder_vector = decoder_vector.squeeze() # Remove batch & seq dims
                # --- End vector computation ---

                # Calculate the change we want to apply.
                delta = strength * decoder_vector
                
                new_hiddens = hidden_states.clone()
                new_hiddens[:, -1, :] += delta.to(original_dtype)

                return (new_hiddens,) + out[1:] if isinstance(out, tuple) else new_hiddens


            hooks.append(layer_to_hook.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                outputs = self.subject_model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                log_probs_next_token = F.log_softmax(next_token_logits, dim=-1) if get_logp_dist else None

                gen_ids = self.subject_model.generate(
                    input_ids, max_new_tokens=self.max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.pad_token_id
                )
            generated_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        finally:
            for h in hooks:
                h.remove()
        
        return generated_text, log_probs_next_token.cpu() if get_logp_dist else torch.empty(0)


    async def _score_explanation(self, generated_text: str, explanation: str) -> float:
        """Computes log P(explanation | generated_text) under the subject model."""
        device = self._get_device()

        # Create the full input sequence: context + explanation
        context_enc = self.tokenizer(generated_text, return_tensors="pt")
        explanation_enc = self.tokenizer(explanation, return_tensors="pt")

        full_input_ids = torch.cat(
            [context_enc.input_ids, explanation_enc.input_ids], dim=1
        ).to(device)

        with torch.no_grad():
            outputs = self.subject_model(full_input_ids)
            logits = outputs.logits

        # We only need to score the explanation part
        context_len = context_enc.input_ids.shape[1]
        # Get logits for positions that predict the explanation tokens
        explanation_logits = logits[:, context_len - 1 : -1, :]

        # Get the target token IDs for the explanation
        target_ids = explanation_enc.input_ids.to(device)

        log_probs = F.log_softmax(explanation_logits, dim=-1)

        # Gather the log-probabilities of the actual explanation tokens
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum().item()


        """
        Retrieves the correct SAE model, handling cases where the framework
        provides a functools.partial wrapper.
        """
        candidate = None
        
        # 1. Try to get the SAE from the record object first.
        if hasattr(record, "sae") and record.sae:
            candidate = record.sae
        # 2. If not on the record, look it up in the explainer_model dictionary.
        elif self.explainer_model and isinstance(self.explainer_model, dict):
            full_key = self._get_full_hookpoint_path(hookpoint_str)
            for key in [hookpoint_str, full_key]:
                if self.explainer_model.get(key) is not None:
                    candidate = self.explainer_model.get(key)
                    break
        
        if candidate is not None:
            # 3. Check if we need to unwrap a partial object.
            if isinstance(candidate, functools.partial):
                # Case A: The instance is in a bound method's __self__.
                instance = getattr(candidate.func, '__self__', None)
                if instance is not None:
                    return instance  # Unwrapped successfully.
                
                # Case B: The instance is the first argument to the partial.
                if candidate.args and len(candidate.args) > 0:
                    instance = candidate.args[0]
                    # A sanity check to make sure it looks like an SAE model.
                    if hasattr(instance, 'encode') and hasattr(instance, 'decode'):
                        return instance  # Unwrapped successfully.
                
                # If we found a partial but failed to unwrap it, we cannot proceed.
                print(f"ERROR: Found a partial for {hookpoint_str} but could not unwrap the SAE instance.")
                return None
            
            # If it's not a partial, it's the model itself.
            return candidate

        print(f"ERROR: Surprisal scorer could not find an SAE for hookpoint '{hookpoint_str}'")
        return None


    def _get_sae_for_hookpoint(self, hookpoint_str: str, record: LatentRecord) -> Any:
        """
        Retrieves the correct SAE model, handling the specific functools.partial
        wrapper provided by the Delphi framework.
        """
        candidate = None
        
        if hasattr(record, "sae") and record.sae:
            candidate = record.sae
        elif self.explainer_model and isinstance(self.explainer_model, dict):
            full_key = self._get_full_hookpoint_path(hookpoint_str)
            for key in [hookpoint_str, full_key]:
                if self.explainer_model.get(key) is not None:
                    candidate = self.explainer_model.get(key)
                    break
        
        if candidate is not None:
            if isinstance(candidate, functools.partial):
                if candidate.keywords and 'sae' in candidate.keywords:
                    return candidate.keywords['sae']
            
            return candidate

        print(f"ERROR: Surprisal scorer could not find an SAE for hookpoint '{hookpoint_str}'")
        return None


    def _get_intervention_direction(self, record: LatentRecord) -> torch.Tensor:
            hookpoint_str = self.hookpoint_str or getattr(record, "hookpoint", None)
            
            sae = self._get_sae_for_hookpoint(hookpoint_str, record)
                
            if sae and hasattr(sae, "get_feature_vector"):
                direction = sae.get_feature_vector(record.feature_id)
                if not isinstance(direction, torch.Tensor):
                    direction = torch.tensor(direction, dtype=torch.float32)
                direction = direction.squeeze()
                return F.normalize(direction, p=2, dim=0)

            return self._estimate_direction_from_examples(record)


    def _estimate_direction_from_examples(self, record: LatentRecord) -> torch.Tensor:
        """Estimates an intervention direction by averaging activations."""
        device = self._get_device()

        examples = self._sanitize_examples(getattr(record, "test", []) or [])
        if not examples:
            hidden_dim = self.subject_model.config.hidden_size
            return torch.zeros(hidden_dim, device=device)

        captured_activations = []

        def capture_hook(module, inp, out):
            hidden_states = out[0] if isinstance(out, tuple) else out

            captured_activations.append(hidden_states[:, -1, :].detach().cpu())

        hookpoint_str = self.hookpoint_str or getattr(record, "hookpoint", None)
        layer_to_hook = self._resolve_hookpoint(self.subject_model, hookpoint_str)
        handle = layer_to_hook.register_forward_hook(capture_hook)

        try:
            for ex in examples[: min(8, self.num_prompts)]:
                prompt = "".join(ex["str_tokens"])
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    device
                )
                with torch.no_grad():
                    self.subject_model(input_ids)
        finally:
            handle.remove()

        if not captured_activations:
            hidden_dim = self.subject_model.config.hidden_size
            return torch.zeros(hidden_dim, device=device)

        activations = torch.cat(captured_activations, dim=0).to(device)
        direction = activations.mean(dim=0)

        return F.normalize(direction, p=2, dim=0)
