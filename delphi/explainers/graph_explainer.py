import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

from delphi.explainers.default.prompts import SYSTEM_GRAPH
from delphi.explainers.explainer import Explainer, ExplainerResult, Response
from delphi.latents.latents import (
    ActivatingExample,
    LatentRecord,
)


@dataclass
class GraphExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""
    cot: bool = False
    """Whether to use chain of thought reasoning in the prompt."""
    max_examples: int = 15
    """Maximum number of activating examples to use."""
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
    disable_thinking: bool = False
    """Appends /no_think to the user prompt to disable thinking on Qwen3 models"""

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

        top_logits = record.top_logits if self.top_logits else []
        bot_logits = record.bot_logits if self.bot_logits else []
        if self.graph_info_path is None or not os.path.exists(self.graph_info_path):
            from delphi import logger

            logger.error("Graph info path is not set.")
            return ExplainerResult(
                record=record,
                explanation="Graph info path was not set to a valid file path.",
            )

        parent_explanations_files = record.parents[
            : min(len(record.parents), self.max_parent_explanations)
        ]
        parent_explanations = []
        for parent in parent_explanations_files:
            parent_path = os.path.join(self.explanations_dir, str(parent))
            if not os.path.exists(parent_path):
                continue
            with open(parent_path, "r") as f:
                parent_explanations.append(f.read())

        # Build the prompt with both types of examples
        messages = self._build_prompt(
            activating_examples,
            top_logits,
            bot_logits,
            self.graph_prompt,
            parent_explanations,
        )
        # Generate the explanation
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )

        try:
            if isinstance(response, Response):
                response_text = response.text
            else:
                response_text = response
            explanation = self._parse_explanation(response_text)
            return ExplainerResult(
                record=record,
                explanation=explanation,
                prompt=messages[1]["content"],
                response=response_text,
            )
        except Exception as e:
            print(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record, explanation=response.text, prompt=messages[1]["content"]
            )

    def _parse_explanation(self, response):
        # Extract explanation from the response text
        explanation = ""
        if "[EXPLANATION]" in response:
            explanation = response.split("[EXPLANATION]")[-1].strip()
        else:
            explanation = response.strip()

        method = 0
        if "[SELECTED METHOD]" in response:
            try:
                method = int(response.split("[SELECTED METHOD]")[-1].strip()[-1])
            except Exception as e:
                method = 0
                print(f"Failed to parse method: {repr(e)} from response: {response}")

        prefixes = ["", "[say] ", "", "[promote] ", "[supress] "]
        explanation = prefixes[method] + explanation
        return explanation

    def _parse_information(self, examples: list[ActivatingExample]):
        # Extract relevant information from activating examples
        activating_tokens = []
        text_after_tokens = []
        plain_examples = []
        for example in examples:
            # find non zero activated tokens
            activated_idxs = torch.where(example.normalized_activations > 0)[0]
            activated_list = activated_idxs.tolist()
            activating_tokens.extend([example.str_tokens[i] for i in activated_list])
            example.str_tokens.append("")  # to avoid index error
            text_after_tokens.extend(
                [example.str_tokens[i + 1] for i in activated_list]
            )

            plain_examples.append(" ".join(example.str_tokens))

        return activating_tokens, text_after_tokens, plain_examples

    def _build_prompt(  # type: ignore
        self,
        examples: list[ActivatingExample],
        top_logits: list[str],
        bot_logits: list[str],
        prompt: str,
        parent_explanations: list[str],
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
        (activating_tokens, text_after_tokens, plain_examples) = (
            self._parse_information(examples)
        )
        highlighted_examples.append("\nMAX_ACTIVATING_TOKENS:")
        highlighted_examples.append(", ".join(activating_tokens))

        highlighted_examples.append("\nTOKENS_AFTER_MAX_ACTIVATING_TOKEN:")
        highlighted_examples.append(", ".join(text_after_tokens))

        highlighted_examples.append("\nTOP_POSITIVE_LOGITS:")
        highlighted_examples.append(", ".join(top_logits))

        highlighted_examples.append("\nTOP_NEGATIVE_LOGITS:")
        highlighted_examples.append(", ".join(bot_logits))

        highlighted_examples.append("\nTOP_ACTIVATING_TEXT:")
        highlighted_examples.append(", ".join(plain_examples))

        # If there are parent explanations, add them to the prompt
        if parent_explanations:
            highlighted_examples.append("\nTOP_PARENT_EXPLANATIONS:")
            highlighted_examples.append(", ".join(parent_explanations))

        # If a prompt is provided, include it in the messages
        if prompt:
            highlighted_examples.append(f"\nGRAPH_PROMPT: \n{prompt}")

        highlighted_examples.append("\n### OUTPUT:")
        highlighted_examples.append("/no_think" if self.disable_thinking else "")
        highlighted_examples_str = "\n".join(highlighted_examples)

        # Create messages array with the system prompt
        return [
            {"role": "system", "content": SYSTEM_GRAPH},
            {
                "role": "user",
                "content": highlighted_examples_str,
            },
        ]

    def call_sync(self, record):
        """Synchronous wrapper for the asynchronous __call__ method."""
        return asyncio.run(self.__call__(record))

    def _log_prompt(self, prompt, feature, output):
        log_entry = {
            "feature": feature,
            "prompt": prompt,
            "output": output,
        }
        with open("prompt_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
