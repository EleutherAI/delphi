import asyncio
import re
from dataclasses import dataclass
from typing import NamedTuple

import torch

from delphi.explainers.contrastive_prompt import prompt
from delphi.explainers.explainer import Explainer, Response
from delphi.latents.latents import ActivatingExample, LatentRecord, NonActivatingExample


class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""

    short_name: str
    """Short name of the latent."""

    confidence: str
    """Confidence of the latent."""


@dataclass
class ContrastiveExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        """
        Override the base __call__ method to use both train and not_active examples.

        Args:
            record: The latent record containing both activating and
                non-activating examples.

        Returns:
            ExplainerResult: The explainer result containing the explanation.
        """
        # Sample from both activating and non-activating examples
        activating_examples = record.train

        non_activating_examples = []
        if len(record.not_active) > 0:
            non_activating_examples = record.not_active

            # Ensure non-activating examples have normalized activations for consistency
            for example in non_activating_examples:
                if example.normalized_activations is None:
                    # Use zeros for non-activating examples
                    example.normalized_activations = torch.zeros_like(
                        example.activations
                    )

        # Build the prompt with both types of examples
        messages = self._build_prompt(activating_examples, non_activating_examples)
        # Generate the explanation
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )

        try:
            if isinstance(response, Response):
                response_text = response.text
            else:
                response_text = response
            explanation, short_name, confidence = self.parse_explanation(response_text)
            if self.verbose:
                from ..logger import logger

                logger.info(f"Explanation: {explanation}")
                logger.info(f"Messages: {messages[-1]['content']}")
                logger.info(f"Response: {response}")

            return ExplainerResult(
                record=record,
                explanation=explanation,
                short_name=short_name,
                confidence=confidence,
            )
        except Exception as e:
            from ..logger import logger

            logger.error(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record,
                explanation="Explanation could not be parsed.",
                short_name="unknown",
                confidence="unknown",
            )

    def parse_explanation(self, text: str) -> list[str]:
        """
        Parse the explanation, short name, and confidence from the response.
        Returns a list of [explanation, short_name, confidence].
        Only fails if explanation cannot be found.
        """
        try:
            # Extract explanation (required)
            explanation_match = re.search(
                r"Concept description: (.*?)(?:\n|$)", text, re.DOTALL
            )
            if not explanation_match:
                return ["Explanation could not be parsed.", "unknown", "unknown"]

            explanation = explanation_match.group(1).strip()

            # Extract short name (optional)
            short_name_match = re.search(r"Short name: (.*?)(?:\n|$)", text)
            short_name = (
                short_name_match.group(1).strip() if short_name_match else "unknown"
            )

            # Extract confidence (optional)
            confidence_match = re.search(r"Confidence: (.*?)(?:\n|$)", text)
            confidence = (
                confidence_match.group(1).strip() if confidence_match else "unknown"
            )

            return [explanation, short_name, confidence]
        except Exception as e:
            from ..logger import logger

            logger.error(f"Explanation parsing regex failed: {repr(e)}")
            return ["Explanation could not be parsed.", "unknown", "unknown"]

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        result = ""
        threshold = max(activations) * self.threshold

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "<<"
                word_activations = []
                while i < len(str_toks) and check(i):
                    result += f"{str_toks[i]}"
                    word_activations.append(activations[i])
                    i += 1
                result += "|"
                for activation in word_activations:
                    result += f"{activation},"
                result = result[:-1]
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)

    def _build_prompt(  # type: ignore
        self,
        activating_examples: list[ActivatingExample],
        non_activating_examples: list[NonActivatingExample],
    ) -> list[dict]:
        """
        Build a prompt with both activating and non-activating examples clearly labeled.

        Args:
            examples: List containing both activating and non-activating examples.

        Returns:
            A list of message dictionaries for the prompt.
        """
        highlighted_examples = []

        # Process activating examples

        highlighted_examples.append("ACTIVATING EXAMPLES:")
        for i, example in enumerate(activating_examples):
            str_toks = example.str_tokens
            activations = example.normalized_activations.tolist()
            assert str_toks is not None
            ex = self._highlight(str_toks, activations).strip()  # .replace("\n", "")
            highlighted_examples.append(f"Example {i}:  {ex}")

            # if self.activations and example.normalized_activations is not None:
            #     normalized_activations = example.normalized_activations.tolist()
            #     highlighted_examples.append(
            #         self._join_activations(
            #             str_toks, activations, normalized_activations
            #         )
            #     )

        # Process non-activating examples
        if non_activating_examples:
            highlighted_examples.append("\n\nNON-ACTIVATING EXAMPLES:")
            for i, example in enumerate(non_activating_examples):
                str_toks = example.str_tokens
                assert str_toks is not None
                activations = example.normalized_activations.tolist()
                # Note: For non-activating examples, the _highlight method won't
                # highlight anything since activation values will be below threshold

                ex = self._highlight(str_toks, activations).strip().replace("\n", "")
                highlighted_examples.append(f"Example {i}:  {ex}")

        # Join all sections into a single string
        highlighted_examples_str = "\n".join(highlighted_examples)

        # Create messages array with the system prompt
        return prompt(highlighted_examples_str)

    def call_sync(self, record):
        """Synchronous wrapper for the asynchronous __call__ method."""
        return asyncio.run(self.__call__(record))
