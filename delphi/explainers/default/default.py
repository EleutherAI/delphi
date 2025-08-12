import asyncio
from dataclasses import dataclass

from ..explainer import ActivatingExample, Explainer
from .prompt_builder import build_prompt
from ...latents import LatentRecord
from delphi.explainers.explainer import Explainer, ExplainerResult, Response
@dataclass
class DefaultExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""
    cot: bool = False
    """Whether to use chain of thought reasoning."""
    top_logits: bool = True
    """Whether to show the top logits for a feature to the explainer"""
    bot_logits: bool = True
    """Whether to show the bottom logits for a feature to the explainer"""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        """
        Override the base __call__ method to use both train examples and logits.

        Args:
            record: The latent record containing both activating and
                non-activating examples.

        Returns:
            ExplainerResult: The explainer result containing the explanation.
        """
        # Build the prompt with examples and logits
        messages = self._build_prompt(record.train, record.top_logits, record.bot_logits)
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

    def _build_prompt(self, examples: list[ActivatingExample], top_logits, bot_logits) -> list[dict]:
        highlighted_examples = []
        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            highlighted_examples.append(self._highlight(str_toks, activations))

            if self.activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            top_logits=top_logits,
            bot_logits=bot_logits,
            activations=self.activations,
            cot=self.cot,
        )

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
