import asyncio
from dataclasses import dataclass

import torch

from delphi.explainers.default.prompts import SYSTEM_CONTRASTIVE
from delphi.explainers.explainer import Explainer, ExplainerResult, Response
from delphi.latents.latents import ActivatingExample, LatentRecord, NonActivatingExample


@dataclass
class ContrastiveExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""
    max_examples: int = 15
    """Maximum number of activating examples to use."""
    max_non_activating: int = 5
    """Maximum number of non-activating examples to use."""

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
        activating_examples = record.train[: self.max_examples]

        non_activating_examples = []
        if len(record.not_active) > 0:
            non_activating_examples = record.not_active[: self.max_non_activating]

            # Ensure non-activating examples have normalized activations for consistency
            for example in non_activating_examples:
                if example.normalized_activations is None:
                    # Use zeros for non-activating examples
                    example.normalized_activations = torch.zeros_like(
                        example.activations
                    )

        # Combine examples for the prompt
        combined_examples = activating_examples + non_activating_examples

        # Build the prompt with both types of examples
        messages = self._build_prompt(combined_examples)
        # Generate the explanation
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )

        try:
            if isinstance(response, Response):
                response_text = response.text
            else:
                response_text = response
            explanation = self.parse_explanation(response_text)
            if self.verbose:
                from delphi import logger

                logger.info(f"Explanation: {explanation}")
                logger.info(f"Messages: {messages[-1]['content']}")
                logger.info(f"Response: {response}")

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            from delphi import logger

            logger.error(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record, explanation="Explanation could not be parsed."
            )

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

        # Create messages array with the system prompt
        return [
            {
                "role": "system",
                "content": SYSTEM_CONTRASTIVE,
            },
            {
                "role": "user",
                "content": highlighted_examples_str,
            },
        ]

    def call_sync(self, record):
        """Synchronous wrapper for the asynchronous __call__ method."""
        return asyncio.run(self.__call__(record))
