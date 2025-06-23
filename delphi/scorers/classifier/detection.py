from ...clients.client import Client
from ...latents import LatentRecord
from ...scorers.scorer import Scorer
from .classifier import Classifier
from .prompts.detection_prompt import prompt as detection_prompt
from .sample import Sample, examples_to_samples


class DetectionScorer(Classifier, Scorer):
    name = "detection"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        n_examples_shown: int = 1,
        log_prob: bool = False,
        temperature: float = 0.0,
        **generation_kwargs,
    ):
        """
        Initialize a DetectionScorer.

        Args:
            client: The client to use for generation.
            tokenizer: The tokenizer used to cache the tokens
            verbose: Whether to print verbose output.
            n_examples_shown: The number of examples to show in the prompt,
                        a larger number can both leak information and make
                        it harder for models to generate anwers in the correct format
            log_prob: Whether to use log probabilities to allow for AUC calculation
            generation_kwargs: Additional generation kwargs
        """
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            **generation_kwargs,
        )

    def prompt(self, examples: str, explanation: str) -> list[dict]:
        return detection_prompt(examples, explanation)

    def _prepare(self, record: LatentRecord) -> list[Sample]:  # type: ignore
        """
        Prepare and shuffle a list of samples for classification.
        """

        if len(record.not_active) > 0:
            samples = examples_to_samples(
                record.not_active,
            )

        else:
            samples = []

        samples.extend(
            examples_to_samples(
                record.test,  # type: ignore
            )
        )

        return samples
