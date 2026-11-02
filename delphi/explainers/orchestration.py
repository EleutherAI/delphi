"""Explanation search on validation examples, followed by one final evaluation."""

import asyncio
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Callable, Literal

from delphi import logger
from delphi.clients.client import Client, Response
from delphi.latents import LatentRecord
from delphi.scorers.scorer import Scorer, ScorerResult

from .default.default import DefaultExplainer
from .explainer import ExplainerResult


def score_value(result: ScorerResult | None) -> float:
    """Use balanced accuracy, overall simulation correlation, or embedding gap."""
    if result is None or not result.score:
        return -math.inf
    samples = result.score
    if hasattr(samples[0], "correct"):
        if any(s.correct is None for s in samples):
            return -math.inf
        groups = [
            [s for s in samples if s.activating == label] for label in (False, True)
        ]
        return (
            fmean(fmean(float(s.correct) for s in group) for group in groups)
            if all(groups)
            else -math.inf
        )
    if hasattr(samples[0], "ev_correlation_score"):
        overall = [s for s in samples if s.distance == 0]
        value = float(overall[0].ev_correlation_score) if overall else -math.inf
        return value if math.isfinite(value) else -math.inf
    if hasattr(samples[0], "similarity"):
        positives = {"".join(e.str_tokens) for e in result.record.test}
        negatives = {"".join(e.str_tokens) for e in result.record.not_active}
        if positives & negatives:
            raise ValueError(
                "Embedding judge cannot distinguish identical positive/negative text"
            )
        pos = [s.similarity for s in samples if s.text in positives]
        neg = [s.similarity for s in samples if s.text in negatives]
        value = fmean(pos) - fmean(neg) if pos and neg else -math.inf
        return value if math.isfinite(value) else -math.inf
    raise ValueError("Unsupported selection scorer output")


@dataclass
class BestOfKOrchestrator:
    client: Client
    scorers_with_paths: list[tuple[Scorer, Path]] = field(default_factory=list)
    num_explanations: int = 5
    judge_scorer_index: int = 0
    num_train_examples: int | None = 20
    temperature: float = 1.0
    threshold: float = 0.3
    activations: bool = True
    is_multishot: bool = True
    verbose: bool = False
    seed: int = 22
    scorer_postprocess: Callable | None = None
    explainer_postprocess: Callable | None = None

    def __post_init__(self):
        if self.num_explanations < 1:
            raise ValueError("num_explanations must be positive")
        if not 0 <= self.judge_scorer_index < len(self.scorers_with_paths):
            raise ValueError("judge_scorer_index must select a configured scorer")
        if self.num_train_examples is not None and self.num_train_examples < 1:
            raise ValueError("num_train_examples must be positive or None")

    def _split(self, record: LatentRecord) -> tuple[LatentRecord, LatentRecord]:
        train_ids = {id(e) for e in record.train}
        if train_ids & {id(e) for e in record.test + record.not_active}:
            raise ValueError("Training and evaluation examples overlap")
        rng = random.Random(self.seed)

        def partition(examples):
            if not examples:
                return [], []
            if len({id(e) for e in examples}) != len(examples):
                raise ValueError("Evaluation examples contain duplicate objects")
            if len(examples) < 2:
                raise ValueError(
                    "Search requires at least two examples per evaluation class"
                )
            examples = list(examples)
            rng.shuffle(examples)
            midpoint = len(examples) // 2
            return examples[:midpoint], examples[midpoint:]

        val_pos, final_pos = partition(record.test)
        val_neg, final_neg = partition(record.not_active)
        if not val_pos:
            raise ValueError("Search requires activating evaluation examples")
        return (
            replace(record, test=val_pos, not_active=val_neg, extra_examples=val_neg),
            replace(
                record, test=final_pos, not_active=final_neg, extra_examples=final_neg
            ),
        )

    def _prompt(self, record: LatentRecord) -> list[dict]:
        examples = list(record.train)
        if (
            self.num_train_examples is not None
            and len(examples) > self.num_train_examples
        ):
            examples = random.Random(self.seed).sample(
                examples, self.num_train_examples
            )
        if not examples:
            raise ValueError("No training examples available")
        return DefaultExplainer(
            self.client, threshold=self.threshold, activations=self.activations
        )._build_prompt(examples)

    async def _generate(self, prompt: list[dict]) -> str:
        response = await self.client.generate(prompt, temperature=self.temperature)
        if not isinstance(response, Response):
            raise ValueError("Explanation client must return a Response")
        explanation = DefaultExplainer(self.client).parse_explanation(response.text)
        if explanation == "Explanation could not be parsed.":
            raise ValueError("Explanation could not be parsed")
        return explanation

    async def _run_scorers(self, record: LatentRecord) -> list[ScorerResult | None]:
        results = []
        # Sequential scoring keeps each result attached to its candidate and
        # avoids concurrently mutating a shared scorer's RNG/generation settings.
        for scorer, _ in self.scorers_with_paths:
            rng = getattr(scorer, "rng", None)
            state = rng.getstate() if isinstance(rng, random.Random) else None
            try:
                results.append(await scorer(record))
            except Exception as exc:
                logger.warning(f"Candidate scorer failed: {exc!r}")
                results.append(None)
            finally:
                if state is not None:
                    rng.setstate(state)
        return results

    async def _finish(self, final: LatentRecord, explanation: str) -> ExplainerResult:
        final = replace(final, explanation=explanation)
        results = await self._run_scorers(final)
        for result, (_, path) in zip(results, self.scorers_with_paths):
            if result is not None and self.scorer_postprocess:
                self.scorer_postprocess(result, score_dir=path)
        result = ExplainerResult(final, explanation)
        if self.explainer_postprocess:
            self.explainer_postprocess(result)
        return result

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        validation, final = self._split(record)
        prompt = self._prompt(record)
        if self.is_multishot:
            # gather preserves input indices even when requests finish out of order.
            candidates = await asyncio.gather(
                *(self._generate(prompt) for _ in range(self.num_explanations)),
                return_exceptions=True,
            )
        else:
            prompt.append(
                {
                    "role": "user",
                    "content": (
                        f"Return {self.num_explanations} distinct explanations. "
                        "Use one [EXPLANATION]: line per final candidate."
                    ),
                }
            )
            response = await self.client.generate(prompt, temperature=self.temperature)
            if not isinstance(response, Response):
                raise ValueError("Explanation client must return a Response")
            candidates = [
                line.split(":", 1)[1].strip()
                for line in response.text.splitlines()
                if line.startswith("[EXPLANATION]:")
            ][-self.num_explanations :]
        best = None
        best_score = -math.inf
        for explanation in candidates:
            if not isinstance(explanation, str) or not explanation:
                continue
            scores = await self._run_scorers(
                replace(validation, explanation=explanation)
            )
            value = score_value(scores[self.judge_scorer_index])
            if value > best_score:
                best, best_score = explanation, value
        if best is None:
            raise ValueError("No candidate received a valid validation score")
        return await self._finish(final, best)


@dataclass
class HillClimbingOrchestrator(BestOfKOrchestrator):
    num_rounds: int = 3
    max_false_positives: int = 20
    max_false_negatives: int = 20
    carryforward_strategy: Literal["best", "last"] = "last"

    def __post_init__(self):
        super().__post_init__()
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be positive")
        if self.carryforward_strategy not in ("best", "last"):
            raise ValueError("carryforward_strategy must be best or last")
        if min(self.max_false_positives, self.max_false_negatives) < 0:
            raise ValueError("Feedback limits must be nonnegative")

    def _feedback(self, result: ScorerResult | None) -> str:
        if result is None or not isinstance(result.score, list):
            return ""
        limits = {False: self.max_false_positives, True: self.max_false_negatives}
        lines = []
        for sample in result.score:
            if getattr(sample, "correct", None) is not False:
                continue
            label = sample.activating
            if limits[label] > 0:
                limits[label] -= 1
                lines.append(
                    f"False {'negative' if label else 'positive'}: "
                    f"{''.join(sample.str_tokens)}"
                )
        return "\n".join(lines)

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        validation, final = self._split(record)
        base_prompt = self._prompt(record)
        prompt = base_prompt
        best = None
        best_score = -math.inf
        last = None
        for _ in range(self.num_rounds):
            explanation = await self._generate(prompt)
            scores = await self._run_scorers(
                replace(validation, explanation=explanation)
            )
            judge = scores[self.judge_scorer_index]
            value = score_value(judge)
            if math.isfinite(value):
                last = explanation
                if value > best_score:
                    best, best_score = explanation, value
            prompt = base_prompt + [
                {
                    "role": "user",
                    "content": (
                        f"Refine this explanation: {explanation}\n"
                        "Validation feedback:\n"
                        f"{self._feedback(judge)}\nReturn a final [EXPLANATION]: line."
                    ),
                }
            ]
        chosen = best if self.carryforward_strategy == "best" else last
        if chosen is None:
            raise ValueError("No round received a valid validation score")
        return await self._finish(final, chosen)
