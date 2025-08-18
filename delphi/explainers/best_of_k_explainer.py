import asyncio
import re
from dataclasses import dataclass
from typing import List

from delphi import logger
from delphi.explainers.default.prompts import SYSTEM_BEST_OF_K_ONESHOT
from delphi.explainers.explainer import Explainer, ExplainerResult, Response
from delphi.latents.latents import ActivatingExample, LatentRecord, NonActivatingExample


@dataclass
class BestOfKExplainer(Explainer):
    num_explanations: int = 3
    """Number of explanations to generate."""
    is_one_shot: bool = True
    """Whether to use a different prompt for each explanation, or just request K"""

    async def __call__(self, record: LatentRecord) -> List[ExplainerResult]:
        """
        Override the base __call__ method to implement the best-of-k explainer.

        Args:
            record: The latent record containing both activating and
                non-activating examples.

        Returns:
            ExplainerResultMulti: The explainer result containing the explanations.
        """
        messages = self._build_prompt(record.train)
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )

        try:
            if isinstance(response, Response):
                response_text = response.text
            else:
                response_text = response
                assert isinstance(response_text, str)

            explanations = self.parse_explanations(response_text)

            if self.verbose:
                logger.info(
                    f"[BestOfKExplainer::__call__] Explanation(s): {explanations}"
                )
                logger.info(
                    f"[BestOfKExplainer::__call__] Messages: {messages[-1]['content']}"
                )
                logger.info(f"[BestOfKExplainer::__call__] Response: {response}")

            return [
                ExplainerResult(record=record, explanation=explanation)
                for explanation in explanations
            ]
            # ExplainerResultMulti(record=record, explanations=explanations)
        except Exception as e:
            logger.error(
                f"[BestOfKExplainer::__call__] Explanation parsing failed: {repr(e)}"
            )
            return [
                ExplainerResult(
                    record=record,
                    explanation="[BestOfKExplainer::__call__] Explanation\
                         could not be parsed.",
                )
            ]

    def parse_explanations(self, text: str) -> List[str]:
        try:
            # Extract as many explanation lines as present, up to K, but return whatever
            # we find
            pattern = re.compile(
                r"^\s*\[\s*EXPLANATION\s*\]\s*:\s*(.+?)\s*$",  # \s* is whitespace
                re.IGNORECASE | re.MULTILINE,  # Multiline makes ^ and $ activate for
                # each line, not just whole string
            )
            matches = pattern.findall(text)
            matches = [m.strip() for m in matches]

            if matches:
                # Return up to the requested number; join to keep return
                # type consistent (str)
                if self.verbose:
                    logger.info(
                        f"[BestOfKExplainer::parse_explanation] Found {len(matches)} \
                        well-formed explanations. Requested {self.num_explanations}"
                    )
                if len(matches) > self.num_explanations:
                    logger.warning(
                        f"[BestOfKExplainer::parse_explanations] Found {len(matches)} \
                            explanations, but requested {self.num_explanations}. \
                                Returning {self.num_explanations} explanations."
                    )
                return matches
            else:
                logger.error(
                    "[BestOfKExplainer::parse_explanations] No explanations found."
                    + "\n[BestOfKExplainer::parse_explanations] Text: {text}"
                )

            return ["[BestOfKExplainer::parse_explanations] No explanations found."]
        except Exception as e:
            logger.error(
                f"[BestOfKExplainer::parse_explanations] Explanation parsing \
                    regex failed: {repr(e)}"
            )
            raise

    def _build_prompt(  # type: ignore
        self, examples: list[ActivatingExample | NonActivatingExample]
    ) -> list[dict]:
        """
        Build a prompt with both activating and non-activating examples clearly labeled.

        Args:
            examples: List containing both activating and non-activating examples.

        Returns:
            A list of message dictionaries for the prompt.
        """
        highlighted_examples = []

        # First, separate activating and non-activating examples
        activating_examples = [
            ex for ex in examples if isinstance(ex, ActivatingExample)
        ]
        non_activating_examples = [
            ex for ex in examples if not isinstance(ex, ActivatingExample)
        ]

        # Process activating examples
        if activating_examples:
            highlighted_examples.append("ACTIVATING EXAMPLES:")
            for i, example in enumerate(activating_examples, 1):
                str_toks = example.str_tokens
                activations = example.activations.tolist()
                ex = self._highlight(str_toks, activations).strip().replace("\n", "")
                highlighted_examples.append(f"Example {i}:  {ex}")

                if self.activations and example.normalized_activations is not None:
                    normalized_activations = example.normalized_activations.tolist()
                    highlighted_examples.append(
                        self._join_activations(
                            str_toks, activations, normalized_activations
                        )
                    )

        # Process non-activating examples
        if non_activating_examples:
            highlighted_examples.append("\nNON-ACTIVATING EXAMPLES:")
            for i, example in enumerate(non_activating_examples, 1):
                str_toks = example.str_tokens
                activations = example.activations.tolist()
                # Note: For non-activating examples, the _highlight method won't
                # highlight anything since activation values will be below threshold
                ex = self._highlight(str_toks, activations).strip().replace("\n", "")
                highlighted_examples.append(f"Example {i}:  {ex}")

        # Join all sections into a single string
        highlighted_examples_str = "\n".join(highlighted_examples)
        num_explanations_str = (
            "\n"
            + f"The number of explanations you are asked to \
                generate is: {self.num_explanations}."
        )
        system_prompt = SYSTEM_BEST_OF_K_ONESHOT + num_explanations_str
        # Create messages array with the system prompt
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": highlighted_examples_str,
            },
        ]

    def call_sync(self, record: LatentRecord) -> List[ExplainerResult]:
        """Synchronous wrapper for the asynchronous __call__ method."""
        return asyncio.run(self.__call__(record))
