from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional

import blobfile as bf
import orjson
import torch
from jaxtyping import Float, Int
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

    locations: Int[Tensor, "n_examples 3"]
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

    tokens: Int[Tensor, "ctx_len"]
    """Tokenized input sequence."""

    activations: Float[Tensor, "ctx_len"]
    """Activation values for the input sequence."""

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

    normalized_activations: Optional[Float[Tensor, "ctx_len"]] = None
    """Activations quantized to integers in [0, 10]."""

    str_tokens: Optional[list[str]] = None
    """Tokenized input sequence as strings."""

    quantile: int = 0
    """The quantile of the activating example."""


@dataclass
class NonActivatingExample(Example):
    """
    An example of a latent that does not activate a model.
    """

    str_tokens: list[str]
    """Tokenized input sequence as strings."""

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
    """Example sequences where the latent activates, assumed to be sorted in
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
    """Frequency of the latent. Number of activations per total number of tokens."""

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
        do_display: bool = True,
        example_source: Literal["examples", "train", "test"] = "examples",
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

        def _to_string(toks, activations: Float[Tensor, "ctx_len"]) -> str:
            """
            Convert tokens and activations to a string.

            Args:
                tokens: The tokenized input sequence.
                activations: The activation values.

            Returns:
                str: The formatted string.
            """
            text_spacing = "0.00em"
            toks = convert_token_array_to_list(toks)
            activations = convert_token_array_to_list(activations)
            inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
            toks = [
                [
                    inverse_vocab[int(t)]
                    .replace("Ġ", " ")
                    .replace("▁", " ")
                    .replace("\n", "\\n")
                    for t in tok
                ]
                for tok in toks
            ]
            highlighted_text = []
            highlighted_text.append(
                """
        <body style="background-color: black; color: white;">
        """
            )
            max_value = max([max(activ) for activ in activations])
            min_value = min([min(activ) for activ in activations])
            # Add color bar
            highlighted_text.append(
                "Token Activations: " + make_colorbar(min_value, max_value)
            )

            highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
            for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
                for act_ind, (a, t) in enumerate(zip(act, tok)):
                    text_color, background_color = value_to_color(
                        a, max_value, min_value
                    )
                    highlighted_text.append(
                        f'<span style="background-color:{background_color};'
                        f'margin-right: {text_spacing}; color:rgb({text_color})"'
                        f">{escape(t)}</span>"
                    )  # noqa: E501
                highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
            highlighted_text = "".join(highlighted_text)
            return highlighted_text

        match example_source:
            case "examples":
                examples = self.examples
            case "train":
                examples = self.train
            case "test":
                examples = [x[0] for x in self.test]
            case _:
                raise ValueError(f"Unknown example source: {example_source}")
        examples = examples[:n]
        strings = _to_string(
            [example.tokens for example in examples],
            [example.activations for example in examples],
        )

        if do_display:
            from IPython.display import HTML, display

            display(HTML(strings))
        else:
            return strings


def make_colorbar(
    min_value,
    max_value,
    white=255,
    red_blue_ness=250,
    positive_threshold=0.01,
    negative_threshold=0.01,
):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if min_value < -negative_threshold:
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value * ratio), 1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'  # noqa: E501
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'  # noqa: E501
    # Do positive
    if max_value > positive_threshold:
        for i in range(1, num_colors + 1):
            ratio = i / (num_colors)
            value = round((max_value * ratio), 1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'  # noqa: E501
    return colorbar


def value_to_color(
    activation,
    max_value,
    min_value,
    white=255,
    red_blue_ness=250,
    positive_threshold=0.01,
    negative_threshold=0.01,
):
    if activation > positive_threshold:
        ratio = activation / max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"
        background_color = f"rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)"  # noqa: E501
    elif activation < -negative_threshold:
        ratio = activation / min_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"
        background_color = f"rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)"  # noqa: E501
    else:
        text_color = "0,0,0"
        background_color = f"rgba({white},{white},{white},1)"
    return text_color, background_color


def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim() == 2:
            array = array.tolist()
        else:
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if isinstance(array[0], int):
            array = [array]
        if isinstance(array[0], torch.Tensor):
            array = [t.tolist() for t in array]
    return array


def escape(t):
    t = (
        t.replace(" ", "&nbsp;")
        .replace("<bos>", "BOS")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return t
