from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import blobfile as bf
import orjson
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class Latent:
    """
    A latent extracted from a model's activations.
    """

    module_name: str
    """The module name associated with the latent."""

    latent_index: int
    """The index of the latent within the module."""

    def __repr__(self) -> str:
        """
        Return a string representation of the latent.

        Returns:
            str: A string representation of the latent.
        """
        return f"{self.module_name}_latent{self.latent_index}"


class ActivationData(NamedTuple):
    """
    Represents the activation data for a latent.
    """

    locations: Float[Tensor, "n_examples 2"]
    """Tensor of latent locations."""

    activations: Float[Tensor, "n_examples"]
    """Tensor of latent activations."""


class LatentData(NamedTuple):
    """
    Represents the output of a TensorBuffer.
    """

    latent: Latent
    """The latent associated with this output."""

    module: str
    """The module associated with this output."""

    activation_data: ActivationData
    """The activation data for this latent."""


@dataclass
class Neighbour:
    distance: float
    latent_index: int


@dataclass
class Example:
    """
    A single example of latent data.
    """

    tokens: Float[Tensor, "ctx_len"]
    """Tokenized input sequence."""

    activations: Float[Tensor, "ctx_len"]
    """Activation values for the input sequence."""

    str_tokens: list[str]
    """Tokenized input sequence as strings."""

    normalized_activations: Optional[Float[Tensor, "ctx_len"]] = None
    """Activations quantized to integers in [0, 10]."""

    @property
    def max_activation(self) -> float:
        """
        Get the maximum activation value.

        Returns:
            float: The maximum activation value.
        """
        return float(self.activations.max())


@dataclass
class ActivatingExample(Example):
    """
    An example of a latent that activates a model.
    """

    quantile: int = 0
    """The quantile of the activating example."""


@dataclass
class NonActivatingExample(Example):
    """
    An example of a latent that does not activate a model.
    """

    distance: float = 0.0
    """
    The distance from the neighbouring latent.
    Defaults to -1.0 if not using neighbours.
    """


@dataclass
class LatentRecord:
    """
    A record of latent data.
    """

    latent: Latent
    """The latent associated with the record."""

    examples: list[ActivatingExample] = field(default_factory=list)
    """Example sequences where the latent activations, assumed to be sorted in
    descending order by max activation."""

    not_active: list[NonActivatingExample] = field(default_factory=list)
    """Non-activating examples."""

    train: list[ActivatingExample] = field(default_factory=list)
    """Training examples."""

    test: list[ActivatingExample] = field(default_factory=list)
    """Test examples."""

    neighbours: list[Neighbour] = field(default_factory=list)
    """Neighbours of the latent."""

    explanation: str = ""
    """Explanation of the latent."""

    extra_examples: Optional[list[Example]] = None
    """Extra examples to include in the record."""

    per_token_frequency: float = 0.0
    """Frequency of the latent. Number of activations per token."""

    per_context_frequency: float = 0.0
    """Frequency of the latent. Number of activations in a context per total
    number of contexts."""

    @property
    def max_activation(self) -> float:
        """
        Get the maximum activation value for the latent.

        Returns:
            float: The maximum activation value.
        """
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples: bool = False):
        """
        Save the latent record to a file.

        Args:
            directory: The directory to save the file in.
            save_examples: Whether to save the examples. Defaults to False.
        """
        path = f"{directory}/{self.latent}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")
            serializable.pop("train")
            serializable.pop("test")

        serializable.pop("latent")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))

    def set_neighbours(
        self,
        neighbours: list[tuple[float, int]],
    ):
        """
        Set the neighbours for the latent record.
        """
        self.neighbours = [
            Neighbour(distance=neighbour[0], latent_index=neighbour[1])
            for neighbour in neighbours
        ]

    def display(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        threshold: float = 0.0,
        n: int = 10,
    ):
        """
        Display the latent record in a formatted string.

        Args:
            tokenizer: The tokenizer to use for decoding.
            threshold: The threshold for highlighting activations.
                Defaults to 0.0.
            n: The number of examples to display. Defaults to 10.

        Returns:
            str: The formatted string.
        """
        from IPython.core.display import HTML, display  # type: ignore

        def _to_string(tokens: list[str], activations: Float[Tensor, "ctx_len"]) -> str:
            """
            Convert tokens and activations to a string.

            Args:
                tokens: The tokenized input sequence.
                activations: The activation values.

            Returns:
                str: The formatted string.
            """
            result = []
            i = 0

            max_act = activations.max()
            _threshold = max_act * threshold

            while i < len(tokens):
                if activations[i] > _threshold:
                    result.append("<mark>")
                    while i < len(tokens) and activations[i] > _threshold:
                        result.append(tokens[i])
                        i += 1
                    result.append("</mark>")
                else:
                    result.append(tokens[i])
                    i += 1
                return "".join(result)
            return ""

        strings = [
            _to_string(tokenizer.batch_decode(example.tokens), example.activations)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))
