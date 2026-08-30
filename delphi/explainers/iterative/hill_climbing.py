"""
Iterative Hill-Climbing Explainer Orchestrator

Generates explanations iteratively, using scorer feedback (false positives/negatives)
to refine explanations over multiple rounds.
"""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Callable, Literal, Optional

import torch

from delphi import logger
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import (
    ActivatingExample,
    Example,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers.scorer import Scorer, ScorerResult

# System prompt for iterative refinement
SYSTEM_ITERATIVE = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.
Your task is to provide a necessary and sufficient explanation that predicts when the pattern is present (i.e., what condition causes the tokens to be marked)

- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- You may be given a previous attempted explanation of the pattern, along with some false-negative or false-positives. Please use these to refine the explanation - do NOT return the same explanation, instead refine it based on the new data.
- If iterating on a given explanation, the examples will be labeled according to type (normal, false-negative, false-positive - e.g. a false-positive example is an example that was incorrectly identified as having the pattern based on the explanation shown.
- If you are not given a prior explanation, examples will not be labeled and are all normal examples known to activate the pattern.
- The last line of your response must be the explanation, beginning with "[EXPLANATION]:" followed by the explanation with no line breaks. Your answer will be processed programmatically so please comply with these rules.

"""


@dataclass
class HillClimbingOrchestrator:
    """
    Iteratively refines explanations using scorer feedback over multiple rounds.

    Each round:
    1. Generate/refine explanation based on examples and previous errors
    2. Score the explanation
    3. Extract false positives/negatives for next round
    4. Repeat

    Final result selected by 'best' (highest score) or 'last' (most recent) strategy.
    """

    client: object
    """LLM client for generating explanations."""

    scorers_with_paths: list[tuple[Scorer, Path]] = field(default_factory=list)
    """List of (scorer, output_path) tuples for evaluation."""

    num_rounds: int = 3
    """Number of refinement rounds."""

    judge_scorer_index: int = 0
    """Index of scorer to use for selecting best explanation."""

    max_false_positives: int = 20
    """Maximum false positive examples to include in refinement prompts."""

    max_false_negatives: int = 20
    """Maximum false negative examples to include in refinement prompts."""

    carryforward_strategy: Literal["best", "last"] = "last"
    """Strategy for final selection: 'best' or 'last'."""

    num_train_examples_per_round: int = 20
    """Number of training examples to show each round."""

    threshold: float = 0.3
    """Activation threshold for highlighting tokens."""

    activations: bool = True
    """Whether to show activation values in prompts."""

    temperature: float = 0.0
    """Sampling temperature for generation."""

    verbose: bool = False
    """Whether to log verbose output."""

    scorer_postprocess: Optional[Callable] = None
    """Postprocessing function for scorer results."""

    explainer_postprocess: Optional[Callable] = None
    """Postprocessing function for explainer results."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        """Run iterative refinement and return best/last explanation."""

        # Split data
        (
            train_pool,
            test_activating,
            test_non_activating,
            holdout_activating,
            holdout_non_activating,
        ) = self._split_train_test_holdout(record)

        explanations = []
        all_holdout_scores = []
        wrong_examples = []
        current_explanation = None

        for round_idx in range(self.num_rounds):
            # Sample training examples
            if len(train_pool) > self.num_train_examples_per_round:
                sampled_train = random.sample(
                    train_pool, self.num_train_examples_per_round
                )
            else:
                sampled_train = train_pool

            # Build prompt
            if current_explanation is None:
                # Initial prompt
                messages = self._build_initial_prompt(sampled_train)
            else:
                # Refinement prompt with FP/FN feedback
                messages = self._build_refinement_prompt(
                    sampled_train, current_explanation, wrong_examples
                )

            # Generate explanation
            response = await self.client.generate(
                messages, temperature=self.temperature
            )
            explanation_text = self._parse_explanation(response.text)

            # Create result
            result_record = LatentRecord(
                latent=record.latent,
                train=sampled_train,
                test=test_activating,
                not_active=test_non_activating,
                explanation=explanation_text,
            )
            result = ExplainerResult(record=result_record, explanation=explanation_text)
            explanations.append(result)

            if self.explainer_postprocess:
                self.explainer_postprocess(result, is_final=False)

            # Score on test set
            test_scorer_results = await self._run_scorers(result_record)

            # Score on holdout set
            holdout_record = LatentRecord(
                latent=record.latent,
                train=sampled_train,
                test=holdout_activating,
                not_active=holdout_non_activating,
                explanation=explanation_text,
            )
            holdout_scorer_results = await self._run_scorers(holdout_record)
            all_holdout_scores.append(holdout_scorer_results)

            # Save intermediate scores
            for scorer_idx, (_, score_dir) in enumerate(self.scorers_with_paths):
                if self.scorer_postprocess and holdout_scorer_results[scorer_idx]:
                    self.scorer_postprocess(
                        holdout_scorer_results[scorer_idx],
                        score_dir=score_dir,
                        round_idx=round_idx,
                    )

            # Extract wrong examples for next round
            wrong_examples = self._extract_wrong_examples(test_scorer_results)

            # Update current explanation for next round
            current_explanation = explanation_text

        # Select final explanation
        if self.carryforward_strategy == "best":
            best_idx = self._select_best_idx(all_holdout_scores)
            final_result = explanations[best_idx]
            final_scores = all_holdout_scores[best_idx]
        else:
            final_result = explanations[-1]
            final_scores = all_holdout_scores[-1]

        # Save final scores
        for scorer_idx, (_, score_dir) in enumerate(self.scorers_with_paths):
            if self.scorer_postprocess and final_scores[scorer_idx]:
                self.scorer_postprocess(
                    final_scores[scorer_idx], score_dir=score_dir, is_final=True
                )

        if self.explainer_postprocess:
            self.explainer_postprocess(final_result, is_final=True)

        return final_result

    def _split_train_test_holdout(self, record: LatentRecord):
        """Split record into train pool, test set, and holdout set."""
        train_pool = list(record.train)
        test_activating = list(record.train)  # Use train for FP/FN collection
        test_non_activating = list(record.not_active)
        holdout_activating = list(record.test)  # Use test as holdout
        holdout_non_activating = list(record.not_active)

        return (
            train_pool,
            test_activating,
            test_non_activating,
            holdout_activating,
            holdout_non_activating,
        )

    def _build_initial_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        """Build initial prompt without prior explanation."""
        highlighted = self._format_examples(examples)
        messages = [{"role": "system", "content": SYSTEM_ITERATIVE}]
        messages.append({"role": "user", "content": f"\n{highlighted}\n"})
        return messages

    def _build_refinement_prompt(
        self,
        examples: list[ActivatingExample],
        current_explanation: str,
        wrong_examples: list[Example],
    ) -> list[dict]:
        """Build refinement prompt with FP/FN feedback."""
        # Format normal examples
        highlighted = self._format_examples(examples)

        # Separate FP/FN
        false_positives = []
        false_negatives = []
        for ex in wrong_examples:
            if ex.activations.max() > 0:
                false_negatives.append(ex)
            else:
                false_positives.append(ex)

        fp_str = self._format_examples(
            false_positives[: self.max_false_positives], show_activations=False
        )
        fn_str = self._format_examples(false_negatives[: self.max_false_negatives])

        messages = [{"role": "system", "content": SYSTEM_ITERATIVE}]
        messages.append(
            {"role": "user", "content": f"Normal examples:\n{highlighted}\n"}
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Current explanation: {current_explanation}\n\n"
                    f"False negatives:\n{fn_str}\n"
                    f"False positives:\n{fp_str}\n"
                ),
            }
        )

        return messages

    def _format_examples(
        self, examples: list[Example], show_activations: bool = True
    ) -> str:
        """Format examples with highlighting."""
        parts = []
        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            acts = example.activations.tolist()
            highlighted = self._highlight(str_toks, acts)
            parts.append(f"Example {i}: {highlighted}")

            if show_activations and self.activations:
                if (
                    hasattr(example, "normalized_activations")
                    and example.normalized_activations is not None
                ):
                    norm_acts = example.normalized_activations.tolist()
                    parts.append(self._join_activations(str_toks, acts, norm_acts))

        return "\n".join(parts)

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        """Highlight tokens above threshold."""
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
        """Format activation values."""
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

    def _parse_explanation(self, text: str) -> str:
        """Parse [EXPLANATION]: from response."""
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return "Explanation could not be parsed."

    async def _run_scorers(self, record: LatentRecord) -> list[Optional[ScorerResult]]:
        """Run all scorers on record."""
        results = []
        for scorer, _ in self.scorers_with_paths:
            try:
                result = await scorer(record)
                results.append(result)
            except Exception as e:
                logger.error(f"Scorer failed: {repr(e)}")
                results.append(None)
        return results

    def _extract_wrong_examples(
        self, scorer_results: list[Optional[ScorerResult]]
    ) -> list[Example]:
        """Extract incorrectly classified examples from scorer results."""
        wrong = []
        for result in scorer_results:
            if result is None or not result.score:
                continue

            for sample in result.score:
                if not hasattr(sample, "correct") or sample.correct:
                    continue

                # Create example from wrong prediction
                if sample.activating:
                    ex = ActivatingExample(
                        tokens=torch.tensor(0),
                        activations=torch.tensor(sample.activations),
                        str_tokens=sample.str_tokens,
                        normalized_activations=torch.tensor(sample.activations),
                    )
                else:
                    ex = NonActivatingExample(
                        tokens=torch.tensor(0),
                        activations=torch.tensor(sample.activations),
                        str_tokens=sample.str_tokens,
                    )

                # Deduplicate
                if not any(w.str_tokens == ex.str_tokens for w in wrong):
                    wrong.append(ex)

        return wrong

    def _select_best_idx(self, all_scores: list[list[Optional[ScorerResult]]]) -> int:
        """Select index of best round based on judge scorer."""
        scores = []
        for round_scores in all_scores:
            if round_scores[self.judge_scorer_index]:
                score = self._compute_score(round_scores[self.judge_scorer_index])
            else:
                score = float("-inf")
            scores.append(score)

        return max(range(len(scores)), key=lambda i: scores[i])

    def _compute_score(self, result: ScorerResult) -> float:
        """Compute score from scorer result."""
        samples = result.score or []
        if not samples:
            return float("-inf")

        if hasattr(samples[0], "similarity"):
            # Embedding scorer
            pos = [s.similarity for s in samples if s.activating]
            neg = [s.similarity for s in samples if not s.activating]
            if not pos or not neg:
                return float("-inf")
            return fmean(pos) - fmean(neg)

        # Classifier scorer (F1)
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
