import functools
import random
import copy
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..scorer import Scorer, ScorerResult
from ...latents import LatentRecord, ActivatingExample

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
         and intervened generated texts: log P(explanation | text)[cite: 209].
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
    
    def _resolve_hookpoint(self, model: Any, hookpoint_str: str) -> Any:
        """
        Dynamically finds the correct model prefix and resolves the full hookpoint path.
        
        This makes the scorer agnostic to different transformer architectures.
        """
        parts = hookpoint_str.split('.')
        
        is_valid_format = (
            len(parts) == 3 and
            parts[0] in ['layers', 'h'] and
            parts[1].isdigit() and
            parts[2] in ['mlp', 'attention', 'attn']
        )

        if not is_valid_format:
            if len(parts) == 1 and hasattr(model, hookpoint_str):
                 return getattr(model, hookpoint_str)
            raise ValueError(f"""Hookpoint string '{hookpoint_str}' is not in a recognized format 
                                 like 'layers.6.mlp'.""")

        #Heuristically find the model prefix.
        prefix = None
        for p in ["gpt_neox", "transformer", "model"]:
            if hasattr(model, p):
                candidate_body = getattr(model, p)
                if hasattr(candidate_body, parts[0]):
                    prefix = p
                    break
        
        full_path = f"{prefix}.{hookpoint_str}" if prefix else hookpoint_str

        try:
            return self._find_layer(model, full_path)
        except AttributeError as e:
            raise AttributeError(f"""Could not resolve path '{full_path}'.
                            Model structure might be unexpected. Original error: {e}""")


    def _sanitize_examples(self, examples: List[Any]) -> List[Dict[str, Any]]:
        """
        Function used for formatting results to run smoothly in the delphi pipeline
        """
        sanitized = []
        for ex in examples:
            if hasattr(ex, 'str_tokens') and ex.str_tokens is not None:
                sanitized.append({'str_tokens': ex.str_tokens})
            
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
            result = SurprisalInterventionResult(score=0.0, avg_kl=0.0, explanation=record_copy.explanation)
            return ScorerResult(record=record, score=result)

        examples = self._sanitize_examples(raw_examples)
        
        record_copy.test = examples
        record_copy.examples = examples
        record_copy.train = examples
        
        prompts = ["".join(ex["str_tokens"]) for ex in examples[:self.num_prompts]]
        
        total_diff = 0.0
        total_kl = 0.0
        n = 0

        for prompt in prompts:
            clean_text, clean_logp_dist = await self._generate_with_and_without_intervention(prompt, record_copy, intervene=False)
            int_text, int_logp_dist = await self._generate_with_and_without_intervention(prompt, record_copy, intervene=True)
            
            logp_clean = await self._score_explanation(clean_text, record_copy.explanation)
            logp_int = await self._score_explanation(int_text, record_copy.explanation)
            
            p_clean = torch.exp(clean_logp_dist)
            kl_div = F.kl_div(int_logp_dist, p_clean, reduction='sum', log_target=False).item()
            
            total_diff += logp_int - logp_clean
            total_kl += kl_div
            n += 1

        avg_diff = total_diff / n if n > 0 else 0.0
        avg_kl = total_kl / n if n > 0 else 0.0
        final_score = avg_diff / (avg_kl + 1e-9) if n > 0 else 0.0

        final_output_list = []
        for ex in examples[:self.num_prompts]:
            final_output_list.append({
                "str_tokens": ex["str_tokens"],
                "final_score": final_score,
                "avg_kl_divergence": avg_kl,
                # Add placeholder keys that the parser expects, with default values.
                "distance": None,
                "activating": None,
                "prediction": None,
                "correct": None,
                "probability": None,
                "activations": None,
            })
        return ScorerResult(record=record_copy, score=final_output_list)

    async def _generate_with_and_without_intervention(
        self, prompt: str, record: LatentRecord, intervene: bool
    ) -> Tuple[str, torch.Tensor]:
        """
        Generates a text continuation and returns the next-token log-probabilities.

        If `intervene` is True, it adds a feature direction to the activations at the
        specified hookpoint before generation.

        Returns:
            A tuple containing:
            - The generated text (string).
            - The log-probability distribution for the token immediately following 
              the prompt (Tensor).
        """
        device = self._get_device()
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        input_ids = enc["input_ids"].to(device)
        
        hooks = []
        if intervene:

            hookpoint_str = self.hookpoint_str or getattr(record, "hookpoint", None)
            if hookpoint_str is None:
                raise ValueError("No hookpoint string specified for intervention.")

            layer_to_hook = self._resolve_hookpoint(self.subject_model, hookpoint_str)

            direction = self._get_intervention_direction(record).to(device)
            direction = direction.unsqueeze(0).unsqueeze(0)  # Shape for broadcasting: [1, 1, D]

            def hook_fn(module, inp, out):
                hidden_states = out[0] if isinstance(out, tuple) else out
                
                # Apply intervention to the last token's hidden state
                hidden_states[:, -1:, :] += self.strength * direction
                
                # Return the modified activations in their original format
                if isinstance(out, tuple):
                    return (hidden_states,) + out[1:]
                return hidden_states

            hooks.append(layer_to_hook.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                # 1. Get next-token logits for KL divergence calculation
                outputs = self.subject_model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                log_probs_next_token = F.log_softmax(next_token_logits, dim=-1)

                # 2. Generate the full text continuation
                gen_ids = self.subject_model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        finally:
            for h in hooks:
                h.remove()

        return generated_text, log_probs_next_token.cpu()

    async def _score_explanation(self, generated_text: str, explanation: str) -> float:
        """Computes log P(explanation | generated_text) under the subject model."""
        device = self._get_device()
        
        # Create the full input sequence: context + explanation
        context_enc = self.tokenizer(generated_text, return_tensors="pt")
        explanation_enc = self.tokenizer(explanation, return_tensors="pt")
        
        full_input_ids = torch.cat([context_enc.input_ids, explanation_enc.input_ids], dim=1).to(device)
        
        with torch.no_grad():
            outputs = self.subject_model(full_input_ids)
            logits = outputs.logits

        # We only need to score the explanation part
        context_len = context_enc.input_ids.shape[1]
        # Get logits for positions that predict the explanation tokens
        explanation_logits = logits[:, context_len - 1:-1, :]
        
        # Get the target token IDs for the explanation
        target_ids = explanation_enc.input_ids.to(device)
        
        log_probs = F.log_softmax(explanation_logits, dim=-1)
        
        # Gather the log-probabilities of the actual explanation tokens
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.sum().item()

    def _get_intervention_direction(self, record: LatentRecord) -> torch.Tensor:
        """
        Gets the feature direction vector, preferring an SAE if available,
        otherwise falling back to estimating it from activations.
        """
        # --- Fast Path: Try to get vector from an SAE-like explainer model ---
        if self.explainer_model:
            sae = None
            candidate = self.explainer_model
            if isinstance(self.explainer_model, dict):
                hookpoint_str = self.hookpoint_str or getattr(record, "hookpoint", None)
                candidate = self.explainer_model.get(hookpoint_str)

            if hasattr(candidate, 'get_feature_vector'):
                sae = candidate
            elif hasattr(candidate, 'sae') and hasattr(candidate.sae, 'get_feature_vector'):
                sae = candidate.sae

            if sae:
                direction = sae.get_feature_vector(record.feature_id)
                if not isinstance(direction, torch.Tensor):
                    direction = torch.tensor(direction, dtype=torch.float32)
                direction = direction.squeeze()
                return F.normalize(direction, p=2, dim=0)

        # --- Fallback: Estimate direction from activating examples ---
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
            for ex in examples[:min(8, self.num_prompts)]:
                prompt = "".join(ex["str_tokens"])
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
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