# Output-based intervention scorer (Gur-Arieh et al. 2025)
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import random
from ...scorer import Scorer, ScorerResult
from ...latents import LatentRecord, ActivatingExample
from transformers import PreTrainedModel

@dataclass
class OutputInterventionResult:
    """Result of output-based intervention evaluation."""
    score: int           # +1 if target set chosen, -1 otherwise
    explanation: str
    example_text: str

class OutputInterventionScorer(Scorer):
    """
    Output-based evaluation by steering (clamping) the feature and using a judge LLM
    to pick which outputs best match the description:contentReference[oaicite:5]{index=5}.
    We generate texts for the target feature and for a few random features, 
    then ask the judge to choose the matching set.
    """
    name = "output_intervention"

    def __init__(self, subject_model: PreTrainedModel, explainer_model, **kwargs):
        self.subject_model = subject_model
        self.explainer_model = explainer_model
        self.steering_strength = kwargs.get("strength", 5.0)
        self.num_prompts = kwargs.get("num_prompts", 3)
        self.num_random = kwargs.get("num_random_features", 2)
        self.hookpoint = kwargs.get("hookpoint", "transformer.h.6.mlp")
        self.tokenizer = getattr(subject_model, "tokenizer", None)

    async def __call__(self, record: LatentRecord) -> ScorerResult:
        # Prepare activating prompts
        examples = [ex for ex in record.test if isinstance(ex, ActivatingExample)]
        random.shuffle(examples)
        prompts = ["".join(str(t) for t in ex.str_tokens) for ex in examples[:self.num_prompts]]

        # Generate text for the target feature
        target_texts = []
        for p in prompts:
            text, _ = await self._generate(p, record.feature_id, self.steering_strength)
            target_texts.append(text)

        # Pick a few random feature IDs (avoid the target)
        random_ids = []
        while len(random_ids) < self.num_random:
            rid = random.randint(0, 999)
            if rid != record.feature_id:
                random_ids.append(rid)

        # Generate texts for random features
        random_sets = []
        for fid in random_ids:
            rand_texts = []
            for p in prompts:
                text, _ = await self._generate(p, fid, self.steering_strength)
                rand_texts.append(text)
            random_sets.append(rand_texts)

        # Create prompt for judge LLM
        judge_prompt = self._format_judge_prompt(record.explanation, target_texts, random_sets)
        judge_response = await self._ask_judge(judge_prompt)

        # Parse judge response: check if target set was chosen
        resp_lower = judge_response.lower()
        if "target" in resp_lower or "set 1" in resp_lower:
            score = 1
        elif "set 2" in resp_lower or "set 3" in resp_lower or "random" in resp_lower:
            score = -1
        else:
            score = 0

        example_text = prompts[0] if prompts else ""
        detailed = OutputInterventionResult(
            score=score,
            explanation=record.explanation,
            example_text=example_text
        )
        return ScorerResult(record=record, score=detailed)

    async def _generate(self, prompt: str, feature_id: int, strength: float):
        """
        Generates text with the feature clamped (added to hidden state). 
        Returns the (partial) generated text and logits.
        """
        tokenizer = self.tokenizer or __import__("transformers").AutoTokenizer.from_pretrained("gpt2")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Forward hook to clamp feature activation
        direction = self.explainer_model.get_feature_vector(feature_id)
        def hook_fn(module, inp, out):
            out[:, -1, :] = out[:, -1, :] + strength * direction.to(out.device)
            return out
        layer = self._find_layer(self.subject_model, self.hookpoint)
        handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = self.subject_model(input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
        handle.remove()

        text = tokenizer.decode(input_ids[0])
        return text, log_probs

    def _format_judge_prompt(self, explanation: str, target_texts: list, other_sets: list):
        """
        Constructs a prompt for the judge LLM listing each set of texts
        under the target feature and random features.
        """
        prompt = f"Feature description: \"{explanation}\"\n"
        prompt += "Which of the following sets of generated texts best matches this description?\n\n"
        prompt += "Set 1 (target feature):\n"
        for txt in target_texts:
            prompt += f"- {txt}\n"
        for i, rand_set in enumerate(other_sets, start=2):
            prompt += f"\nSet {i} (random feature):\n"
            for txt in rand_set:
                prompt += f"- {txt}\n"
        prompt += "\nAnswer (mention the set number or 'target'/'random'): "
        return prompt

    async def _ask_judge(self, prompt: str) -> str:
        """
        Queries a judge LLM (e.g., GPT-4) with the prompt. Stubbed here.
        """
        # TODO: Implement actual LLM call to get response
        return ""

    def _find_layer(self, model, name: str):
        """Locate a module by its dotted name."""
        current = model
        for attr in name.split('.'):
            if attr.isdigit():
                current = current[int(attr)]
            else:
                current = getattr(current, attr)
        return current
