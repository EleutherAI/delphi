import traceback
from typing import List

import dspy
from dspy.utils.exceptions import AdapterParseError
from pydantic import BaseModel

from delphi.clients.client import Client
from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import LatentRecord
from delphi.logger import logger


class DSPyExplainer:
    name = "default"

    def __init__(
        self,
        client: Client,
        threshold: float = 0.3,
        cot: bool = False,
        few_shot: bool = True,
        **generation_kwargs,
    ):
        self.client = client
        self.generation_kwargs = generation_kwargs
        self.threshold = threshold
        client = self.client.client
        dspy.configure(lm=client)
        self.explainer = (dspy.ChainOfThought if cot else dspy.Predict)(
            ExplanationSignature
        )
        if few_shot:
            # We currently have our own few-shot examples, might want to
            # instead try to optimize the few-shot examples to get better results
            self.explainer = dspy.LabeledFewShot().compile(
                self.explainer,
                trainset=TRAINSET,
            )

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        records = record.train
        all_examples = []
        for ex in records:
            assert ex.str_tokens is not None
            assert ex.normalized_activations is not None
            text = "".join(ex.str_tokens)
            # get activating tokens
            activations = ex.normalized_activations.tolist()
            # find the activations above the threshold
            max_activation = ex.max_activation
            non_zero = [
                (t, a)
                for a, t in zip(activations, ex.str_tokens)
                if a > self.threshold * max_activation
            ]
            all_examples.append(
                FeatureExample(text=text, tokens_and_activations=non_zero)
            )

        try:
            result = await self.explainer.acall(feature_examples=all_examples)
        except AdapterParseError:
            logger.error("Error in explainer:")
            traceback.print_exc()
            result = ExplanationSignature(
                feature_examples=all_examples,
                shared_features=[""],
                hypothesis="",
                explanation="Explanation could not be parsed.",
            )
        return ExplainerResult(record, result.explanation)


class FeatureExample(BaseModel):
    """A text snippet that is an example that activates a feature."""

    text: str
    tokens_and_activations: list[tuple[str, float]]


class ExplanationSignature(dspy.Signature):
    """
    As an AI researcher, your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
    Guidelines:

    You will be given a list of text examples, each with a list of tokens that activated a classifier,
    - Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
    - If the examples are uninformative, you don't need to mention them.
    - Try to summarize the patterns found in the examples, mentioning some of the tokens where the pattern is likely to be found.
    - Don't give over-general explanations.

    """

    feature_examples: List[FeatureExample] = dspy.InputField(
        desc="A list of examples that activate the feature."
    )
    # special_words: List[str] = dspy.OutputField(desc="Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.")
    shared_features: List[str] = dspy.OutputField(
        desc="Write down general shared features of the text examples. This could be related to the full sentence or to the words surrounding the marked words."
    )
    hypothesis: str = dspy.OutputField(desc="Formulate an hypothesis.")
    explanation: str = dspy.OutputField(
        desc="A concise single-sentence explanation of the feature. This is the final output of the model and is what will be graded."
    )


TRAINSET = [
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text="and he was over the moon to find",
                tokens_and_activations=[("over", 5), ("the", 6), ("moon", 9)],
            ),
            FeatureExample(
                text="it was better than meets the eye",
                tokens_and_activations=[
                    ("than", 4),
                    ("meets", 7),
                    ("the", 2),
                    ("eye", 5),
                ],
            ),
            FeatureExample(
                text="I was also over the moon when",
                tokens_and_activations=[("over", 6), ("the", 5), ("moon", 8)],
            ),
            FeatureExample(
                text="it's more than meets the eye",
                tokens_and_activations=[
                    ("than", 3),
                    ("meets", 6),
                    ("the", 1),
                    ("eye", 4),
                ],
            ),
        ],
        shared_features=[
            "The examples contain common idioms.",
            "The activating tokens are parts of common idioms.",
            "The text examples all convey positive sentiment.",
        ],
        hypothesis="The activation values are the highest for the more common idioms.",
        explanation="Common idioms in text conveying positive sentiment.",
    ),
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text="a river is wide but the ocean is wider. The ocean",
                tokens_and_activations=[("er", 8)],
            ),
            FeatureExample(
                text='every year you get taller," she',
                tokens_and_activations=[("er", 2)],
            ),
            FeatureExample(
                text="the hole was smaller but deeper than the",
                tokens_and_activations=[("er", 9), ("er", 9)],
            ),
        ],
        shared_features=[
            "The activating token appeared at the end of a comparative adjective.",
            "The comparative adjectives describe size.",
        ],
        hypothesis="The activation values are higher when comparing physical sizes more directly.",
        explanation='The token "er" at the end of a comparative adjective describing size.',
    ),
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text='something happening inside my house", he',
                tokens_and_activations=[("house", 7)],
            ),
            FeatureExample(
                text='presumably was always contained in a box", according',
                tokens_and_activations=[("a", 5), ("box", 9)],
            ),
            FeatureExample(
                text='people were coming into the smoking area". However he',
                tokens_and_activations=[("smoking", 2), ("area", 4)],
            ),
            FeatureExample(
                text='Patrick: "why are you getting in the way?" Later,',
                tokens_and_activations=[("way", 4), ("?", 2)],
            ),
        ],
        shared_features=[
            "The activating tokens are things that one can be in (literally or figuratively).",
            "The activating token is followed by a quotation mark, suggesting it occurs within speech.",
        ],
        hypothesis="The activation values are highest for distinctive objects or spaces.",
        explanation="Nouns representing distinct objects that contain something, often preceding a quotation mark.",
    ),
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text="John gave her the book",
                tokens_and_activations=[("gave", 7)],
            ),
            FeatureExample(
                text="Monica was talking to her friend",
                tokens_and_activations=[("talking", 5), ("to", 6)],
            ),
            FeatureExample(
                text="I was talking to Maria and I think she did not like it",
                tokens_and_activations=[("think", 5)],
            ),
            FeatureExample(
                text='"Her acting in this movie was was so great, I didn"t know she was so good',
                tokens_and_activations=[("know", 5)],
            ),
        ],
        shared_features=[
            "The activating tokens didn't have a clear pattern of activation.",
            "The activating tokens are always before a female pronoun.",
        ],
        hypothesis="The activation values are highest on places where the next word would be a female pronoun.",
        explanation="Tokens preceding a female pronoun.",
    ),
]
