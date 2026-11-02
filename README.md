# Introduction

Delphi was the home of a temple to Phoebus Apollo, which famously had the inscription, 'Know Thyself.' This library lets language models know themselves through automated interpretability.

This library provides utilities for generating and scoring text explanations of sparse autoencoder (SAE) and transcoder features. The explainer and scorer models can run locally with vLLM, through OpenRouter, or against an external OpenAI-compatible chat server.

The branch used for the article [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/pdf/2410.13928) is the legacy branch [article_version](https://github.com/EleutherAI/delphi/tree/article_version), that branch contains the scripts to reproduce our experiments. Note that we're still actively improving the codebase and that the newest version on the main branch could require slightly different usage.

# Installation

Install this library as a local editable installation. Run the following command from the `delphi` directory.

```bash
pip install -e .
# Include plotting dependencies for the CLI's summary visualizations:
pip install -e ".[visualize]"
```

# Getting Started

To run the default pipeline from the command line, use the following command:

`python -m delphi EleutherAI/pythia-160m EleutherAI/Pythia-160m-SST-k32-32k --n_tokens 10_000_000 --max_latents 100 --hookpoints layers.5.mlp --scorers detection --filter_bos --name pythia-example`

This command will:
1. Cache activations for the first 10 million tokens of the default dataset, `EleutherAI/SmolLM2-135M-10B`.
2. Generate explanations for the first 100 features of layer 5 using the default explainer model, `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`.
3. Score the explanations using the detection scorer.
4. Log summary metrics including per-scorer F1 scores and confusion matrices, and produce histograms of the scorer classification accuracies.

The pipeline is highly configurable and can also be called programmatically (see the [end-to-end test](https://github.com/EleutherAI/delphi/blob/main/tests/e2e.py) for an example).

To use experimental features, create a custom pipeline. You can take inspiration from the main pipeline in [delphi.\_\_main\_\_](https://github.com/EleutherAI/delphi/blob/main/delphi/__main__.py).

## Caching

The first step to generate explanations is to cache sparse model activations. To do so, load your sparse models into the base model, load the tokens you want to cache the activations from, create a `LatentCache` object and run it. We recommend caching over at least 10M tokens.

```python
from datasets import load_dataset
from sparsify.data import chunk_and_tokenize
from delphi.latents import LatentCache

data = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train[:1%]")
tokens = chunk_and_tokenize(data, tokenizer, max_seq_len=256, text_key="raw_content")["input_ids"]

cache = LatentCache(
    model,
    submodule_dict,
    batch_size = 8
)

cache.run(n_tokens = 10_000_000, tokens=tokens)
```

See `populate_cache` in `delphi.__main__` for a full example. Both Sparsify sparse coders and Gemma sparse coders are supported.

Caching saves `.safetensors` of `dict["activations", "locations", "tokens"]`.

```python
from pathlib import Path

cache.save_splits(
    n_splits=5,
    save_dir=Path("raw_latents")
)
```

Safetensors are split into shards over the width of the autoencoder when saved; this is not incremental streaming of activations to disk.

## Loading Latent Records

The `.latents` module provides utilities for reconstructing and sampling various statistics for sparse features. The `LatentDataset` will construct lazy loaded buffers that load activations into memory when called as an iterator object. For ease of use with the autointerp pipeline, we have a *constructor* and *sampler*: the constructor defines builds the context windows from the cached activations and tokens, and the sampler divides these contexts into a training and testing set, used to generate explanations and evaluate them.

```python
from delphi.latents import LatentDataset
from delphi.config import SamplerConfig, ConstructorConfig


latent_dict = {
    ".model.layer.0": torch.arange(0, 131072)
}
sampler_cfg = SamplerConfig()
constructor_cfg = ConstructorConfig()

dataset = LatentDataset(
    raw_dir="feature_folder",
    modules=[".model.layer.0"], # This a list of the different caches to load from
    sampler_cfg=sampler_cfg,
    constructor_cfg=constructor_cfg,
    latents=latent_dict,
    tokenizer=tokenizer
)
```

### Sampling and context windows

`SamplerConfig.train_type` supports `top`, `random`, `quantiles`, and `mix`.
Test examples are sampled from quantiles with the explanation-training examples
excluded. Requested counts need not be divisible by the quantile count; a short
request is distributed across quantiles rather than dropped. A quantile with
insufficient available examples can still return fewer than requested.

`ConstructorConfig.non_activating_source` supports `random`, `neighbours`, and
`FAISS`. Neighbour selection can use co-occurrence, decoder similarity, or encoder
similarity. The main CLI constructs the neighbour cache when needed; custom
loaders must load the appropriate neighbour data as well.

Windows are uncentered by default (`center_examples=False`). Set
`center_examples=True` for peak-aligned windows. Explicit centering introduces
a position cue for simulation scoring. The centering flag was inverted before
v0.2.0: invert an explicit value in an old configuration to reproduce its old
behavior.

### FAISS Index for Hard Negatives

When constructing features for explanation, you can use FAISS (semantic similarity search) to create hard negative examples. Hard negatives are non-activating examples that are semantically similar to activating examples. This approach:

1. Creates embeddings for both activating and non-activating examples using the specified embedding model
2. Builds a FAISS index for efficient similarity search
3. Finds non-activating examples that are semantically similar to activating examples
4. Optionally caches embeddings to speed up future runs

To use FAISS for hard negatives, set the `non_activating_source` parameter to "FAISS" in your `ConstructorConfig`:

```python
from delphi.config import ConstructorConfig

constructor_cfg = ConstructorConfig(
    non_activating_source="FAISS",
    faiss_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    faiss_embedding_cache_enabled=True,
    faiss_embedding_cache_dir=".embedding_cache"
)
```

### Contrastive Explainer

The `ContrastiveExplainer` adds both positive (activating) and negative (non-activating) examples to a single explainer prompt so the explainer model is less likely to label features that are not exclusive to the feature activations (ie we are more likely to provide non activating tokens which are semantically similar). This explainer is automatically used when the `non_activating_source` is set to "FAISS".

```python
from delphi.explainers import ContrastiveExplainer

explainer = ContrastiveExplainer(
    client,
    threshold=0.3,
    max_examples=15,
    max_non_activating=5,
    verbose=True
)
```

## Generating Explanations

Choose an in-process vLLM client, OpenRouter, or an external chat server, then create an explainer from `delphi.explainers`.

For in-process vLLM:

```python
from delphi.clients import Offline

client = Offline(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_memory=0.8, max_model_len=5120, num_gpus=1,
)
```

For OpenRouter:

```python
import os
from delphi.clients import OpenRouter

client = OpenRouter(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

For an already running chat server:

```python
from delphi.clients import ServerClient

client = ServerClient(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1/",
)
```

After choosing one client:

```python
from delphi.explainers import DefaultExplainer

# LatentDataset supplies the decoded tokens; this constructor needs no tokenizer.
explainer = DefaultExplainer(client)
```

The explainer should be added to a pipe, which will send the explanation requests to the client. The pipe should have a function that happens after the request is completed, to e.g. save the data, and could also have a function that happens before the request is sent, e.g to transform some of the data.

```python
from delphi.pipeline import process_wrapper

def explainer_postprocess(result):

    with open(f"{explanation_dir}/{result.record.latent}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))

    return result

explainer_pipe = process_wrapper(explainer,
    postprocess=explainer_postprocess,
)
```
The pipe should then be used in a pipeline. Running the pipeline sends requests concurrently, up to `max_concurrent`.

```python
from delphi.pipeline import Pipeline
import asyncio

pipeline = Pipeline(
    loader,
    explainer_pipe,
)

asyncio.run(pipeline.run(max_concurrent=10))
```

### Explainer availability

| Class | CLI selection | Purpose |
| --- | --- | --- |
| `DefaultExplainer` | `--explainer default` | Generate an explanation from activating training examples. |
| `ContrastiveExplainer` | Automatic when `--non_activating_source FAISS` is used with the default explainer | Show activating and hard-negative examples together. |
| `BestOfKOrchestrator` | `--explainer bestofk` | Generate candidates and select by validation score. |
| `HillClimbingOrchestrator` | `--explainer iterative` | Refine an explanation using validation feedback. |
| `NoOpExplainer` / saved explanation loading | `--explainer none` | Reuse explanations already saved in the run's explanations directory. |
| `SingleTokenExplainer` | Python API only | Use the single-token explanation prompt. |

For best-of-K, use `--bestofk_num_explanations`,
`--bestofk_judge_scorer_index`, and `--bestofk_num_train_examples`.
For iterative search, use `--iterative_num_rounds`,
`--iterative_max_false_positives`, `--iterative_max_false_negatives`, and
`--iterative_carryforward_strategy best` (or `last`). Both modes accept
`--search_temperature` and use the configured `--scorers`.

Search uses training examples for generation, splits the evaluation examples
into validation and final subsets, and saves scores from evaluating the selected
explanation on the final subset once. Validation data drive selection and
feedback; final data do not. Each nonempty evaluation class needs at least two
examples to make this split. Search is applied to the record as a whole; the
standard pipeline's `max_concurrent` is not a candidate-search concurrency setting.

For custom pipelines, `explanation_loader` reads a feature's saved explanation;
`random_explanation_loader` samples a saved explanation from a directory. Random
sampling is not a guarantee that every feature receives a different explanation,
so enforce that separately when constructing a strict shuffled-explanation control.

### External servers

The CLI accepts `--server_port 8000` for localhost or
`--server_url http://host:8000/v1/` for a remote API. Use one of these options,
with `--explainer_provider offline` (the default), and set `--explainer_model`
to the server's served model name. Optional authentication uses `SERVER_API_KEY`.
OpenRouter instead uses `--explainer_provider openrouter` and
`OPENROUTER_API_KEY`.

External-server generation does not initialize a local vLLM model, although
subject-model activation caching still needs its own model. External chat servers
use JSON simulation; the local Offline client uses prompt-logprob simulation.
Close a programmatically created `ServerClient` with `await client.close()`
when finished. The CLI handles its cleanup.

## Scoring and feature evaluation

The package exports the following scorers. **Only `detection`, `fuzz`, and
`simulation` are currently selectable with the CLI's `--scorers` option.**
The other scorers require a custom Python pipeline or a direct call.

| Scorer | CLI name | What it measures |
| --- | --- | --- |
| [`DetectionScorer`](delphi/scorers/classifier/detection.py) | `detection` | Whether an example fits the explanation. |
| [`FuzzingScorer`](delphi/scorers/classifier/fuzz.py) | `fuzz` | Whether marked tokens fit the explanation. |
| [`OpenAISimulator`](delphi/scorers/simulator/simulation/oai_simulator.py) | `simulation` | Correlation between predicted and observed token activations. |
| [`IntruderScorer`](delphi/scorers/classifier/intruder.py) | API only | Find the odd example among related activating examples, without reading an explanation. |
| [`EmbeddingScorer`](delphi/scorers/embedding/embedding.py) | API only | Embed an explanation and compare it with example embeddings. |
| [`ExampleEmbeddingScorer`](delphi/scorers/embedding/example_embedding.py) | API only | Compare positive/negative example groups through embedding similarity, without an explanation. |
| [`SurprisalScorer`](delphi/scorers/surprisal/surprisal.py) | API only | Compare per-token language-model losses with explanation and baseline prefixes. |

`EmbeddingScorer` and `ExampleEmbeddingScorer` take a compatible
`sentence_transformers` model; `SurprisalScorer` takes a language model with a
`.tokenizer` attribute. They do not use a vLLM generation client. Surprisal here
is prefix-conditioned loss scoring, not feature-intervention scoring.

### Intruder detection

Intruder detection is an **explanation-free feature-coherence test**, rather
than a test of a supplied explanation. `type="default"` inserts a non-activating
example among activating examples from one activation quantile. `type="internal"`
instead inserts an activating example from a different quantile.

```python
from delphi.scorers import IntruderScorer

# dataset is a configured LatentDataset; client is a generation client.
record = next(iter(dataset))
scorer = IntruderScorer(client, n_examples_shown=4, type="default")
result = scorer.call_sync(record)  # Or: result = await scorer(record)

assert result.score, "No intruder groups were constructed"
accuracy = sum(item.correct for item in result.score) / len(result.score)
```

The current constructor defaults to a one-example group, so set
`n_examples_shown` explicitly to at least 2; use enough activating examples
per quantile for the desired group size. Both modes currently require
`record.not_active`, and internal mode also needs at least two activation
quantiles. Results contain the predicted and correct intruder indices, a
correctness flag, and the model's interpretation. These outputs have a different
schema from binary detection/fuzzing results: aggregate them separately rather
than passing them to the CLI's classification summary logger.

### Fuzzing protocol

`--fuzz_type default` uses target-inactive windows. `--fuzz_type active` uses
activating windows whose marked tokens have zero target activation. Each
retained positive mark set is transferred to one negative of the same token
length, matching the observed distribution of marked positions and counts
between labels. Outputs save `highlighted_indices` for auditing. Unmatched pairs
are omitted with a coverage warning; a record with no compatible pairs fails
explicitly. Scores from this protocol should not be compared directly with the
older fixed-position/count protocol. Shuffled-explanation controls remain useful.

### Composing a scoring pipeline

Scorers consume `LatentRecord` objects with `record.explanation` set. Explainers
return `ExplainerResult`, so extract its record before scoring. Multiple scorers
can share a `Pipe`:

```python
import asyncio
from functools import partial
from pathlib import Path

import orjson

from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer


def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    record.extra_examples = record.not_active
    return record


def scorer_postprocess(result, score_dir):
    score_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(result.record.latent).replace("/", "--")
    (score_dir / f"{safe_name}.txt").write_bytes(orjson.dumps(result.score))
    return result


scorer_pipe = Pipe(
    process_wrapper(
        DetectionScorer(client, n_examples_shown=1),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=Path("scores/detection")),
    ),
    process_wrapper(
        FuzzingScorer(client, n_examples_shown=1),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=Path("scores/fuzz")),
    ),
)

pipeline = Pipeline(dataset, explainer_pipe, scorer_pipe)
asyncio.run(pipeline.run(max_concurrent=10))
```

`DetectionScorer` replaces the obsolete `RecallScorer` name found in older
examples. The current detection/fuzzing constructors do not need `tokenizer` or
`batch_size` arguments. Their prompt grouping is controlled by
`n_examples_shown`; the CLI equivalent is `--num_examples_per_scorer_prompt`.
`--log_probs` requests binary token probabilities where the client supports them.

## Inspecting records and rerunning stages

`LatentRecord.display(tokenizer, example_source="test", do_display=False)` returns
HTML for test examples; `example_source` also accepts `examples` and `train`.
For an existing run, `--overwrite cache`, `--overwrite neighbours`, or
`--overwrite scores` selects stages to recompute. Results live under
`results/<name>/latents`, `explanations`, `scores`, `neighbours`, and `visualize`;
`run_config.json` records the configuration.

## Examples

Current examples are in [`examples/`](examples), including
[`latent_contexts_visualization.py`](examples/latent_contexts_visualization.py).
For the complete CLI pipeline, see [`delphi/__main__.py`](delphi/__main__.py);
for a programmatic run, see [`tests/e2e.py`](tests/e2e.py).

## Experiments

The experiments discussed in [the blog post](https://blog.eleuther.ai/autointerp/) were mostly run in a legacy version of this code, which can be found in the [Experiments](https://github.com/EleutherAI/delphi/tree/Experiments) branch.

## Development

Set up the pre-commit lint and run the unit tests:

```bash
pip install -e ".[dev,visualize]"
pre-commit install
pytest
```

From the repository checkout, run the end-to-end script:

```bash
python -m tests.e2e
```

The end-to-end script downloads models and a dataset and performs GPU inference;
it is separate from the ordinary unit-test run. Use `python -m delphi --help`
for the complete generated CLI option list.

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for releases.

## License

Copyright 2024 the EleutherAI Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
