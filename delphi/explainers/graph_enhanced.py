import asyncio
from collections import defaultdict
from dataclasses import dataclass
import os
import json
import math
from typing import Optional, List, Tuple, Dict

import torch

from delphi.explainers.default.prompts import SYSTEM_GRAPH, GRAPH_PROMPT, TOP_LOGITS, BOT_LOGITS, PARENT_NODE_PROMPT, GRAPH_COT
from delphi.explainers.explainer import Explainer, ExplainerResult, Response
from delphi.latents.latents import ActivatingExample, LatentRecord, NonActivatingExample, Latent


@dataclass
class GraphExplainer(Explainer):
    activations: bool = False
    """Whether to show activations to the explainer."""
    cot: bool = False
    """Whether to use chain of thought reasoning in the prompt."""
    max_examples: int = 15
    """Maximum number of activating examples to use."""
    max_non_activating: int = 5
    """Maximum number of non-activating examples to use."""
    graph_info_path: Optional[os.PathLike] = None
    """Path to the graph information file."""
    explanations_dir: Optional[os.PathLike] = None
    """Path to the directory where explanations will be saved."""
    graph_prompt: str = ""
    """The prompt used to generate the graph."""
    max_parent_explanations: int = 1
    """Maximimum number of explanations from parent nodes to use"""
    top_logits: bool = True
    """Whether to show the top logits for a feature to the explainer"""
    bot_logits: bool = True
    """Whether to show the bottom logits for a feature to the explainer"""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        """
        Override the base __call__ method to use
        - train examples
        - non activating examples
        - prompt used to generate graph
        - explanations from the parent nodes

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

        top_logits = record.top_logits if self.top_logits else []
        bot_logits = record.bot_logits if self.bot_logits else []
        if self.graph_info_path is None or not os.path.exists(self.graph_info_path):
            from delphi import logger

            logger.error("Graph info path is not set.")
            return ExplainerResult(
                record=record, explanation="Graph info path was not set to a valid file path."
            )
        
        parent_explanations_files = record.parents[:min(len(record.parents), self.max_parent_explanations)]
        parent_explanations = []
        for parent, influence in parent_explanations_files:

            parent_path = os.path.join(self.explanations_dir, str(parent))
            if not os.path.exists(parent_path):
                print(f"Parent explanation file does not exist: {parent_path}")
                continue
            with open(parent_path, "r") as f:
                parent_explanations.append((f.read(), influence))

        # Build the prompt with both types of examples
        messages = self._build_prompt(combined_examples, top_logits, bot_logits, self.graph_prompt, parent_explanations)
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
            print(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record, explanation=repr(e)
            )

    def _build_prompt(  # type: ignore
        self, examples: list[ActivatingExample | NonActivatingExample],
        top_logits: list[str],
        bot_logits: list[str],
        prompt: str,
        parent_explanations: list[str]
    ) -> list[dict]:
        """
        Build a prompt with graph information

        Args:
            examples: List containing both activating and non-activating examples.
            top_logits: List of top logits to include in the prompt.
            bot_logits: List of bottom logits to include in the prompt.
            prompt: The prompt to include in the message.
            parent_explanations: List of explanations from parent nodes.

        Returns:
            A list of message dictionaries for the prompt.
        """
        highlighted_examples = ["### INPUTS:"]

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

        # If there are parent explanations, add them to the prompt
        if parent_explanations:
            highlighted_examples.append("\nPARENT EXPLANATIONS:")
            for i, (explanation, strength) in enumerate(parent_explanations, 1):
                highlighted_examples.append(f"Parent Explanation {i}: {explanation} (Strength: {strength})")

        # If a prompt is provided, include it in the messages
        if prompt:
            highlighted_examples.append(f"\nPROMPT: {prompt}")
        # If top logits are provided, include them in the messages
        if top_logits:
            highlighted_examples.append(f"\nTOP LOGITS: {', '.join(top_logits)}")
        # If bottom logits are provided, include them in the messages
        if bot_logits:
            highlighted_examples.append(f"\nBOTTOM LOGITS: {', '.join(bot_logits)}")

        highlighted_examples.append("\n### Now output the explanation as requested:")
        # Join all sections into a single string
        highlighted_examples_str = "\n".join(highlighted_examples)
        with open("prompt_log.txt", "w+") as f:
            f.write(highlighted_examples_str)

        # build the system prompt
        graph_prompt = GRAPH_PROMPT if self.graph_prompt else ""
        top_logits_prompt = TOP_LOGITS if self.top_logits else ""
        bot_logits_prompt = BOT_LOGITS if self.bot_logits else ""
        parent_explanations_prompt = PARENT_NODE_PROMPT if self.max_parent_explanations > 0 else ""
        cot_prompt = GRAPH_COT if self.cot else ""

        # Create messages array with the system prompt
        return [
            {
                "role": "system",
                "content": SYSTEM_GRAPH.format(
                    graph_prompt=graph_prompt,
                    top_logits=top_logits_prompt,
                    bot_logits=bot_logits_prompt,
                    parent_explanations=parent_explanations_prompt,
                    cot=cot_prompt
                ),
            },
            {
                "role": "user",
                "content": highlighted_examples_str,
            },
        ]

    def call_sync(self, record):
        """Synchronous wrapper for the asynchronous __call__ method."""
        return asyncio.run(self.__call__(record))
