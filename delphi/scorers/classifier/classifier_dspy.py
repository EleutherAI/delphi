import random
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import dspy
from dspy.utils.exceptions import AdapterParseError
from pydantic import field_validator

from delphi.logger import logger

from ...clients.client import Client
from ...latents import LatentRecord
from ..scorer import ScorerResult


@dataclass
class FeatureExample:
    """A text snippet that might activate a feature."""

    text: str
    activating: bool
    correct: Optional[bool] = None


class DspyClassifier:
    name = ""

    def __init__(
        self,
        scorer: dspy.Predict,
    ):
        self.scorer = scorer

    @abstractmethod
    def _prepare_and_batch(self, record: LatentRecord) -> list[list[FeatureExample]]:
        pass

    @abstractmethod
    def _check_correct(self, example: FeatureExample, is_feature: bool) -> bool:
        pass

    async def __call__(self, record: LatentRecord):
        batched = self._prepare_and_batch(record)
        results = []
        for batch in batched:
            try:
                result = await self.scorer.acall(
                    feature_description=record.explanation, feature_examples=batch
                )
            except AdapterParseError or ValueError:
                logger.error("Error in explainer:")
                result = None
            if result is not None:
                for example, is_feature in zip(batch, result.is_feature):
                    example.correct = self._check_correct(example, is_feature)
            results.append(example)
        return ScorerResult(record, results)


class DspyFuzzer(DspyClassifier):
    name = "fuzzing"

    def __init__(
        self,
        client: Client,
        n_examples_shown: int = 5,
        seed: int = 42,
        fuzz_type: Literal["default", "active"] = "default",
    ):
        self.client = client
        self.n_examples_shown = n_examples_shown
        client = self.client.client
        dspy.configure(lm=client)
        self.scorer = (dspy.Predict)(ExampleDetector)
        self.scorer = dspy.LabeledFewShot().compile(
            self.scorer,
            trainset=FEW_SHOT,
        )
        self.rng = random.Random(seed)
        self.fuzz_type = fuzz_type
        super().__init__(self.scorer)

    def _highlight_tokens(
        self,
        tokens: list[str],
        activations: list[float],
        positions: list[tuple[int, int]] = [],
    ) -> tuple[str, list[tuple[int, int]]]:
        output = ""
        # branch if highlighting positions are provided
        if len(positions) == 0:
            starts, ends = zip(*positions)
            for i, token in enumerate(tokens):
                if i in starts:
                    output += f"<"
                elif i in ends:
                    output += f">"
                else:
                    output += token
        # branch if using activations to highlight tokens
        else:
            positions = []
            i = 0
            while i < len(tokens):
                if activations[i] > 0:
                    start = i
                    output += f"<"

                    while i < len(tokens) and activations[i] > 0:
                        output += tokens[i]
                        i += 1
                    end = i
                    positions.append((start, end))

                    output += f">"
                else:
                    output += tokens[i]
                    i += 1

        return output, positions

    def _perturb_positions(
        self, positions: list[tuple[int, int]], length: int
    ) -> list[tuple[int, int]]:
        new_positions = []
        for position in positions:
            max_len = length - 1

            if self.rng.random() < 0.5:
                # move one forward if possible
                if position[1] < max_len:
                    new_positions.append((position[0] + 1, position[1] + 1))
                else:
                    new_positions.append((position[0] - 1, position[1] - 1))
            else:
                # move one backward if possible
                if position[0] > 0:
                    new_positions.append((position[0] - 1, position[1] - 1))
                else:
                    new_positions.append((position[0] + 1, position[1] + 1))
            new_positions.append(position)
        return new_positions

    def _prepare_and_batch(self, record: LatentRecord) -> list[list[FeatureExample]]:
        activating_examples = record.test
        non_activating_examples = record.not_active
        all_examples = []
        activating_positions = []
        for example in activating_examples:
            # making type checking happy
            assert example.str_tokens is not None
            text, positions = self._highlight_tokens(
                example.str_tokens, example.activations.tolist()
            )
            all_examples.append(
                FeatureExample(text=text, activating=True, correct=None)
            )
            activating_positions.append(positions)
        if self.fuzz_type == "default":
            for example in non_activating_examples:
                assert example.str_tokens is not None

                if sum(example.activations.tolist()) == 0:
                    # sample one of activating positions if example is not activating
                    random_position = self.rng.choice(activating_positions)
                else:
                    # if from a neighbour, we highlight the activating example
                    random_position = []

                text, positions = self._highlight_tokens(
                    example.str_tokens, example.activations.tolist(), random_position
                )
                all_examples.append(
                    FeatureExample(text=text, activating=False, correct=None)
                )
        elif self.fuzz_type == "active":
            for example in activating_examples:
                assert example.str_tokens is not None
                random_position = self.rng.choice(activating_positions)
                # perturb the positions
                new_positions = self._perturb_positions(
                    random_position, len(example.str_tokens)
                )
                text, positions = self._highlight_tokens(
                    example.str_tokens, example.activations.tolist(), new_positions
                )
                all_examples.append(
                    FeatureExample(text=text, activating=False, correct=None)
                )

        self.rng.shuffle(all_examples)

        batched = [
            all_examples[i : i + self.n_examples_shown]
            for i in range(0, len(all_examples), self.n_examples_shown)
        ]

        return batched

    def _check_correct(self, example: FeatureExample, is_feature: bool) -> bool:
        return is_feature == example.activating


class ExampleDetector(dspy.Signature):
    """You are an intelligent and meticulous linguistics researcher.

    You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment".

    You will be given a few examples of text that might contain this feature.

    Some words of the text might be highlighted, these are the tokens that are part of the feature.

    Your task is to determine the examples that are correctly labeled by the given description. Consider that all provided examples could be correct, none of the examples could be correct, or a mix.

    For each example in turn, return true if the sentence is correctly labeled or false if the tokens are mislabeled. You must return your response in a valid Python list. Never output None.
    """

    feature_description: str = dspy.InputField(desc="Feature explanation")
    feature_examples: List[Union[str, FeatureExample]] = dspy.InputField(
        desc="Test examples"
    )
    is_feature: List[Literal[0, 1]] = dspy.OutputField(
        desc="Whether the example is correctly labeled"
    )

    @field_validator("is_feature", mode="after")
    @classmethod
    def check_length(cls, v, info):
        if len(v) != len(info.data["feature_examples"]):
            raise ValueError(
                "Length of is_feature and feature_examples must be the same"
            )
        return v


FEW_SHOT = [
    dspy.Example(
        feature_description="Words related to American football positions, specifically the tight end position.",
        feature_examples=[
            "<|endoftext|>Getty Images Patriots <tight end> Rob Gronkowski had his boss",
            "names of months used in The <Lord> of the Rings: the",
            "Media Day 2015 LSU <defensive end> Isaiah Washington (94) speaks to the",
            "shown, is generally not <eligible for ads>. For example, videos about recent tragedies,",
            "line, with the <left side>, namely tackle Byron Bell at tackle and guard Amini",
        ],
        is_feature=[True, False, True, False, False],
        # is_feature_probabilities=[0.95, 0.05, 0.85, 0.1, 0.75]
    ),
    dspy.Example(
        feature_description='The word "guys" in the phrase "you guys".',
        feature_examples=[
            "enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both",
            "birth control access but I assure you women in Kentucky aren't <laughing> as they struggle",
            "du Soleil Fall Protection Program with construction requirements that do not apply to <theater> settings because",
            "distasteful. Amidst the <slime> lurk bits of Schadenfre",
            "the I want to remind you <guys> that 10 days ago (director Massimil",
        ],
        is_feature=[False, False, False, False, True],
        # is_feature_probabilities=[0.1, 0.1, 0.1, 0.1, 0.1]
    ),
    dspy.Example(
        feature_description='"of" before words that start with a capital letter.',
        feature_examples=[
            "climate, Tomblin's Chief <of> Staff Charlie Lorensen said.",
            "no wonderworking relics, no true Body and <Blood> of Christ, no true Baptism",
            "Deborah Sathe, Head of <Talent> Development and Production at Film London,",
            "It has been devised by Director <of> Public Prosecutions (DPP)",
            "and fair investigation not even include the Director <of> Athletics? Finally, we believe the",
        ],
        is_feature=[True, False, False, True, True],
        # is_feature_probabilities=[0.9, 0.95, 0.9, 0.95, 0.95]
    ),
]
