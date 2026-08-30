"""
Best-of-K Explainer Orchestrator

Generates multiple explanation candidates and selects the best one based on scorer feedback.
This is a standalone orchestrator that manages its own pipeline internally.
"""

import asyncio
import random
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Callable, Optional

from delphi import logger
from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import ActivatingExample, LatentRecord
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers.scorer import Scorer, ScorerResult

from .default.prompt_builder import build_prompt

# System prompt for generating multiple explanations in one shot
SYSTEM_BESTOFK_ONESHOT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- You will be given a number telling you how many explanations you are to generate - these explanations should be meaningfully distinct attempts to encapsulate the pattern. They should not be too similar.
- The final part of your response must consist exclusively of formatted explanations, each explanation on a new line, starting with "[EXPLANATION]:" followed by the explanation. It is imperative that you follow this format as the text is to be processed programmatically.

"""


@dataclass
class BestOfKOrchestrator:
    """
    Generates K explanation candidates and selects the best based on scorer feedback.

    Unlike regular explainers, this orchestrator runs scorers internally to evaluate
    candidates before returning the final result.
    """

    client: object
    """LLM client for generating explanations."""

    scorers_with_paths: list[tuple[Scorer, Path]] = field(default_factory=list)
    """List of (scorer, output_path) tuples for evaluation."""

    num_explanations: int = 5
    """Number of explanation candidates to generate."""

    judge_scorer_index: int = 0
    """Index of scorer to use for selecting best explanation."""

    num_train_examples: Optional[int] = 20
    """Number of training examples to show. None uses all available."""

    temperature: float = 0.0
    """Sampling temperature for generation."""

    threshold: float = 0.3
    """Activation threshold for highlighting tokens."""

    activations: bool = True
    """Whether to show activation values in prompts."""

    is_multishot: bool = True
    """If True, make K separate calls. If False, request K explanations in one call."""

    verbose: bool = False
    """Whether to log verbose output."""

    scorer_preprocess: Optional[Callable] = None
    """Preprocessing function applied before scoring."""

    scorer_postprocess: Optional[Callable] = None
    """Postprocessing function applied after scoring."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        """Generate K explanations, score them, and return the best."""

        # Split into train/test pools
        train_pool, test_activating, test_non_activating = self._split_train_test(
            record
        )

        # Create clean record for scoring
        clean_record = LatentRecord(
            latent=record.latent,
            train=train_pool,
            test=test_activating,
            not_active=test_non_activating,
            explanation=record.explanation,
        )

        # Build prompt and generate explanations
        messages = self._build_prompt(clean_record.train)

        if self.is_multishot:
            # Make K separate calls
            tasks = [
                self.client.generate(messages, temperature=self.temperature)
                for _ in range(self.num_explanations)
            ]
            responses = await asyncio.gather(*tasks)
            combined_text = "\n".join([r.text for r in responses])
            explanations = self._parse_multiple_explanations(combined_text)
        else:
            # Single call requesting K explanations
            oneshot_messages = self._build_oneshot_prompt(clean_record.train)
            response = await self.client.generate(
                oneshot_messages, temperature=self.temperature
            )
            explanations = self._parse_multiple_explanations(response.text)

        # Cap at requested number
        explanations = explanations[: self.num_explanations]

        # Create ExplainerResult for each candidate
        explainer_results = []
        for idx, explanation in enumerate(explanations):
            result_record = LatentRecord(
                latent=clean_record.latent,
                train=clean_record.train,
                test=clean_record.test,
                not_active=clean_record.not_active,
                explanation=explanation,
            )
            explainer_results.append(
                ExplainerResult(record=result_record, explanation=explanation)
            )

        if not explainer_results:
            # Fallback if parsing failed
            return ExplainerResult(
                record=clean_record, explanation="Explanation could not be parsed."
            )

        # Score all candidates
        scorer_results = await self._run_scorers(explainer_results)

        # Select best based on judge scorer
        judge_results = [
            s[self.judge_scorer_index]
            for s in scorer_results
            if s[self.judge_scorer_index]
        ]
        best_idx = self._select_best_idx(judge_results)

        # Save best scores
        for scorer_idx, (scorer, score_dir) in enumerate(self.scorers_with_paths):
            if best_idx < len(scorer_results) and scorer_results[best_idx][scorer_idx]:
                best_score = scorer_results[best_idx][scorer_idx]
                if self.scorer_postprocess:
                    self.scorer_postprocess(best_score, score_dir=score_dir)

        return explainer_results[best_idx]

    def _split_train_test(self, record: LatentRecord) -> tuple[list, list, list]:
        """Split record into train pool, test activating, and test non-activating."""
        return (
            list(record.train),
            list(record.test),
            list(record.not_active),
        )

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        """Build prompt from examples using upstream DefaultExplainer logic."""
        # Sample if needed
        if self.num_train_examples and len(examples) > self.num_train_examples:
            examples = random.sample(examples, self.num_train_examples)

        # Highlight examples
        highlighted = []
        for example in examples:
            str_toks = example.str_tokens
            activations_list = example.activations.tolist()
            highlighted.append(self._highlight(str_toks, activations_list))

            if self.activations and example.normalized_activations is not None:
                normalized = example.normalized_activations.tolist()
                highlighted.append(
                    self._join_activations(str_toks, activations_list, normalized)
                )

        highlighted_str = "\n".join(highlighted)
        return build_prompt(examples=highlighted_str, activations=self.activations)

    def _build_oneshot_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        """Build prompt for one-shot multi-explanation generation."""
        if self.num_train_examples and len(examples) > self.num_train_examples:
            examples = random.sample(examples, self.num_train_examples)

        highlighted = []
        for example in examples:
            str_toks = example.str_tokens
            activations_list = example.activations.tolist()
            highlighted.append(self._highlight(str_toks, activations_list))

            if self.activations and example.normalized_activations is not None:
                normalized = example.normalized_activations.tolist()
                highlighted.append(
                    self._join_activations(str_toks, activations_list, normalized)
                )

        highlighted_str = "\n".join(highlighted)

        messages = [{"role": "system", "content": SYSTEM_BESTOFK_ONESHOT}]
        messages.append({"role": "user", "content": f"\n{highlighted_str}\n"})
        messages.append(
            {
                "role": "user",
                "content": f"The number of explanations to generate is: {self.num_explanations}.",
            }
        )

        return messages

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        """Highlight tokens above threshold with << >> markers."""
        result = ""
        threshold_val = max(activations) * self.threshold if activations else 0

        i = 0
        while i < len(str_toks):
            if activations[i] > threshold_val:
                result += "<<"
                while i < len(str_toks) and activations[i] > threshold_val:
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return result

    def _join_activations(
        self,
        str_toks: list[str],
        token_activations: list[float],
        normalized_activations: list[float],
    ) -> str:
        """Format activation values for display."""
        acts = ""
        count = 0
        threshold_val = (
            max(token_activations) * self.threshold if token_activations else 0
        )

        for str_tok, tok_act, norm_act in zip(
            str_toks, token_activations, normalized_activations
        ):
            if tok_act > threshold_val:
                if count > 10:
                    break
                acts += f'("{str_tok}" : {int(norm_act)}), '
                count += 1

        return "Activations: " + acts

    def _parse_multiple_explanations(self, text: str) -> list[str]:
        """Parse multiple [EXPLANATION]: markers from text."""
        try:
            matches = re.findall(
                r"\[EXPLANATION\]:\s*(.*?)(?=\[EXPLANATION\]:|$)", text, re.DOTALL
            )
            if matches:
                cleaned = [m.strip() for m in matches if m.strip()]
                return cleaned if cleaned else ["Explanation could not be parsed."]
            return ["Explanation could not be parsed."]
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return ["Explanation could not be parsed."]

    async def _run_scorers(
        self, explainer_results: list[ExplainerResult]
    ) -> list[list[Optional[ScorerResult]]]:
        """Run all scorers on all explanation candidates."""
        num_scorers = len(self.scorers_with_paths)
        all_results: list[list[Optional[ScorerResult]]] = [
            [None] * num_scorers for _ in range(len(explainer_results))
        ]

        def make_wrapper(scorer_idx: int):
            scorer, score_dir = self.scorers_with_paths[scorer_idx]
            return process_wrapper(
                scorer,
                preprocess=self.scorer_preprocess,
                postprocess=(
                    partial(
                        self.scorer_postprocess or (lambda r, **_: r),
                        score_dir=score_dir,
                    )
                    if self.scorer_postprocess
                    else None
                ),
            )

        wrappers = [make_wrapper(idx) for idx in range(num_scorers)]

        async def generator():
            for result in explainer_results:
                yield result

        pipeline = Pipeline(generator(), Pipe(*wrappers))
        subset_results = await pipeline.run()

        for pos, scorer_list in enumerate(subset_results):
            for scorer_idx, scorer_result in enumerate(scorer_list):
                all_results[pos][scorer_idx] = scorer_result

        return all_results

    def _select_best_idx(self, scorer_results: list[ScorerResult]) -> int:
        """Select index of best explanation based on scorer results."""
        if not scorer_results:
            return 0

        scores = []
        for result in scorer_results:
            if result is None:
                scores.append(float("-inf"))
                continue
            scores.append(self._compute_score(result))

        return max(range(len(scores)), key=lambda i: scores[i])

    def _compute_score(self, result: ScorerResult) -> float:
        """Compute score from scorer result (F1 for classifier, similarity for embedding)."""
        samples = result.score or []
        if not samples:
            return float("-inf")

        # Check if embedding scorer (has 'similarity' attribute)
        if hasattr(samples[0], "similarity"):
            return self._compute_embedding_score(samples)

        # Otherwise assume classifier output
        return self._compute_f1_score(samples)

    def _compute_f1_score(self, samples) -> float:
        """Compute F1 score from classifier outputs."""
        tp = fp = fn = 0
        for sample in samples:
            if sample.correct:
                if sample.activating:
                    tp += 1
            else:
                if sample.activating:
                    fn += 1
                else:
                    fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    def _compute_embedding_score(self, samples) -> float:
        """Compute embedding score as difference of positive/negative similarities."""
        pos = [s.similarity for s in samples if s.activating]
        neg = [s.similarity for s in samples if not s.activating]
        if not pos or not neg:
            return float("-inf")
        return fmean(pos) - fmean(neg)
