# CHANGELOG


## v0.2.0 (2026-11-02)

### Bug Fixes

- Keep explanation search scores aligned and reserve final evaluation data
  ([`f288c2d`](https://github.com/EleutherAI/delphi/commit/f288c2de4c6503d94c94cf0b863b46dbdf95060c))

- Report classification metrics without optional firing counts
  ([`99599e2`](https://github.com/EleutherAI/delphi/commit/99599e2d74399df6389cf77c45aaa8c7233b8f72))

- Spread the quantile sample remainder instead of dropping it
  ([`c8b0199`](https://github.com/EleutherAI/delphi/commit/c8b019935e7da3e284b7a107f05aad45a7a4cd76))

`split_quantiles` took `n_samples // n_quantiles` from every quantile, so a request that didn't
  divide evenly came back short and one smaller than `n_quantiles` came back empty.
  `--n_examples_test 5` with the default `--n_quantiles 10` asked for five scoring examples and
  produced none, with no warning: the per-quantile count was 0, which is never fewer than the
  examples available, so the "fewer than" log never fired either.

Each quantile's count now comes from the difference of two cumulative counts, which sums to
  `n_samples` exactly and puts the remainder in quantiles spaced across the range rather than
  bunched at the top. A request that already divided evenly, every default among them, takes the
  same examples it did before.

### Features

- Add --server-port option to connect to external vLLM server
  ([`ab02aed`](https://github.com/EleutherAI/delphi/commit/ab02aed9dac54681ff5ecfb87cf20ed0c5afc2e9))

When set, uses OpenAI-compatible API to talk to a vLLM server instead of loading the model
  in-process. Default behavior unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Support external chat servers with explicit simulation capabilities
  ([`f207027`](https://github.com/EleutherAI/delphi/commit/f207027ef60dd485105c9c949259d193f309749e))

- Wire validated explanation search into the CLI
  ([`1cddb36`](https://github.com/EleutherAI/delphi/commit/1cddb36b10a8c1f7271463e744b2c40988328b1c))


## v0.1.4 (2026-11-02)

### Bug Fixes

- Correct centering and sample distinct negative windows
  ([`2bb63f6`](https://github.com/EleutherAI/delphi/commit/2bb63f6b39f0a33346e794271810697c3a960e13))

- Decode str_tokens per token for transformers >= 5
  ([`aed0e64`](https://github.com/EleutherAI/delphi/commit/aed0e64a4df4ee35349890df6e669976b6fc3dda))

On transformers >= 5, batch_decode treats a 1-D tensor as a single sequence and returns one joined
  string instead of per-token strings. This silently broke token highlighting in the explainer and
  classifier scorers and raised IndexError in the intruder scorer.

Add decode_per_token to delphi.utils and use it at the four sites that need per-token strings
  (samplers, non-activating constructor, OpenAI simulator). Fixes #176

- Failed_count is always zero in result analysis
  ([`55b758b`](https://github.com/EleutherAI/delphi/commit/55b758b99f47d29de9d7956d3799d3587e23fc89))

Signed-off-by: WatchTree-19 <119982314+WatchTree-19@users.noreply.github.com>

- Intruderscorer(type="internal") crashes when picking the intruder quantile
  ([`54bd81d`](https://github.com/EleutherAI/delphi/commit/54bd81da6fa9e410af4fb9d69c022982dd0ba3ea))

The internal intruder type removed the source quantile from the list of quantiles with list.remove,
  which compares the quantile lists for equality. Lists of the same length are compared element by
  element, and ActivatingExample's dataclass __eq__ compares its tensors, which raises 'Boolean
  value of Tensor with more than one value is ambiguous'. With the default sampler every quantile
  holds the same number of examples, so every batch whose quantile is not the first one crashed.

Select the other quantiles by their key instead.

- Isolate explainer test imports and respect intruder distance type
  ([`ea5e77f`](https://github.com/EleutherAI/delphi/commit/ea5e77f27d06ec8132e47624b3fdb963fd22fd7f))

- Match fuzzing mark distributions between labels
  ([`a8638f7`](https://github.com/EleutherAI/delphi/commit/a8638f7bf88b71138abcd1127c7e82dbb1d79af0))

- Normalize integer simulator activations under runtime checking
  ([`b65af01`](https://github.com/EleutherAI/delphi/commit/b65af018f150e1ea54383c1d47b67b1ef6e32616))

- Per-module aggregation in result analysis with several hookpoints
  ([`91c30d5`](https://github.com/EleutherAI/delphi/commit/91c30d509a9cd971e9524023f585c0e0d127ea35))

- load_data globbed '*{module}*', so with hookpoints 'layers.1' and 'layers.10' the score files of
  layers.10 were also loaded, and counted, as layers.1. Match '{module}_latent*', the score file
  name. - frequency_weighted_f1 indexed the firing-count tensor with a pandas Series. Torch reads it
  positionally (series[0]) while pandas looks up labels, so every module after the first raised
  KeyError: 0 and log_results crashed for runs with several hookpoints. - The firing count was set
  only when 'latent_idx in counts[module]', a test on the tensor's values; check the index range
  instead.

- Preserve latent records when saving and displaying examples
  ([`9b0702f`](https://github.com/EleutherAI/delphi/commit/9b0702fe0a6f1d9d9dcd7d719bc601ebf8f0273c))

- Preserve positional classifier output compatibility
  ([`4454cf9`](https://github.com/EleutherAI/delphi/commit/4454cf9798896be2dd35d5dffd0d113137c7be61))

- Sample scoring examples from those not shown to the explainer
  ([`93c8bb8`](https://github.com/EleutherAI/delphi/commit/93c8bb8f58b6fa890f015daab16ddaa50fe6f6d0))

sampler() draws the explainer's train examples and the scorers' test examples from the same
  record.examples. With the default config both use split_quantiles, which reseeds the global RNG
  with the same seed, so in quantile 0 random.sample(q, 4) and random.sample(q, 5) make the same
  first 4 picks: 4 of the 5 top-quantile test examples are always train examples. In total 10-20 of
  the 50 default test examples had been shown to the explainer, which inflates detection and fuzzing
  scores.

Exclude the train examples when sampling the test examples, within the original quantile bins.

- Simulator returns zeros on transformers >= 5 and BOS-less tokenizers
  ([`00f33b8`](https://github.com/EleutherAI/delphi/commit/00f33b8cde6cfa342d26b3be92ec361c6bb0edec))

The logprob simulator indexed the result of apply_chat_template as a list of token ids and took
  encode('\t')[1] as the tab token, assuming a BOS token. On transformers >= 5, apply_chat_template
  returns a BatchEncoding by default, and tokenizers such as Qwen2.5 add no BOS, so both raised
  IndexError. simulate() catches the exception and returns all-zero predictions, so every simulation
  correlation became NaN without an error.

Request a list with return_dict=False and encode the tab without special tokens. The offline client
  also passed apply_chat_template's result to TokensPrompt as token ids; request a list there too.

- Surprisalscorer reuses a mutated KV cache across batches
  ([`099c9d1`](https://github.com/EleutherAI/delphi/commit/099c9d1ded4d8aa452b5940fb417b742430550da))

compute_loss_with_kv_cache built the explanation's KV cache once and passed it to every batch, but
  the forward pass appends the batch's tokens to the DynamicCache in place, so later batches
  attended to earlier batches' sentences and got wrong losses. A partial last batch also
  repeat_interleaved an already repeated tensor and failed with a batch-size mismatch. Build the
  explanation cache for each batch.

- Validate classifier answers and score partial batches
  ([`7c55156`](https://github.com/EleutherAI/delphi/commit/7c55156b9ab4be85b4e515d5bbc5cbbc6eabc41f))

- **explainer**: Harden parse_explanation against CoT echo and marker injection
  ([`f04180c`](https://github.com/EleutherAI/delphi/commit/f04180cf4a95fc702e3dfd26870862dc18d3fd75))

parse_explanation used re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL) — first-match with
  greedy-to-EOT capture. Two failure modes:

1. CoT echo (parsing defect): when the explainer reasoned in chain-of-thought and wrote
  [EXPLANATION]: while thinking (considering then rejecting a hypothesis), greedy .* with re.DOTALL
  swallowed the entire response from the first marker to end-of-text. The label became the whole
  reasoning chain rather than the final verdict.

2. Marker injection (security): first-match binding let an early [EXPLANATION]: token win over the
  explainer's actual verdict. The highlighted examples in the explainer's prompt (top-activating
  text from the subject model) are injected verbatim, so a subject model whose top-activating text
  contains '[EXPLANATION]: <benign>' can cause the explainer to echo it, and the echoed marker wins.
  The resulting label flows verbatim into DetectionScorer / IntruderScorer and operator-facing
  tooling (Neuronpedia auto-interp).

Fix: last-match binding with non-greedy capture (no re.DOTALL). Each

[EXPLANATION]: marker binds to only its own line; taking the last match selects the explainer's
  final verdict. Defeats both CoT echo and early marker injection; preserves clean single-marker
  parsing.

Residual: if an injected marker is the literal last line, last-match still binds to it. The complete
  defense (response-aware scrub of any marker appearing in the highlighted-examples input) is a
  follow-up; this PR closes the common cases.

Adds 7 regression tests covering clean parsing, CoT echo, last-match ordering, marker injection, and
  edge cases.

### Documentation

- Clarify release history and remove duplicate changelog entries
  ([`f8da27e`](https://github.com/EleutherAI/delphi/commit/f8da27e7f732b421f4f128e494f7a443d5b900fa))

### Testing

- Read only tensor cache splits and compare original tokens
  ([`4429341`](https://github.com/EleutherAI/delphi/commit/44293414aba443892d4adbd684924de9f81ae4f9))


## v0.1.3 (2026-03-12)

### Bug Fixes

- **classifier**: Make log_prob an explicit constructor option
  ([`fb3aa86`](https://github.com/EleutherAI/delphi/commit/fb3aa863487683285a04f39f3c3134aa6cddb956))

### Continuous Integration

- Add uv quality gates and test discovery config
  ([`f851189`](https://github.com/EleutherAI/delphi/commit/f851189d1f61b2886176e37babc84c3198eb5a32))


## v0.1.2 (2026-03-12)

### Bug Fixes

- Add max_memory CLI support and fix vLLM prompt syntax
  ([#156](https://github.com/EleutherAI/delphi/pull/156),
  [`8a79bb7`](https://github.com/EleutherAI/delphi/commit/8a79bb77f5dbe815e087bb017bddfb8adfc23d4f))

* Add max_memory parameter to run config

Co-authored-by: Simon Schrader <simonschrader96@gmail.com>

* Use configurable max_memory for offline explainer

* Fix breaking change in prompt input formatting from vLLM

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

* Update vllm dependency version to after API breaking change

PR #18800: https://github.com/vllm-project/vllm/releases

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>

- Bugs in _prepare_text and IntruderScorer prompt formatting
  ([#153](https://github.com/EleutherAI/delphi/pull/153),
  [`87f934d`](https://github.com/EleutherAI/delphi/commit/87f934d1b3728b1b6c038398a35287e0e24e4ad7))

* clean up some code and fix bug with preparing text with multiple false positives

* clean up intruderscorer a bit more

* another cleanup in _prepare_and_batch

* cleanup in Pipeline

* fix inconsistency with intruder example formatting

* simplify _generate error reporting

* ignore uv.lock

* undo code readability/style changes

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.1.1 (2025-08-27)

### Bug Fixes

- Make simulator workable
  ([`0ceaa10`](https://github.com/EleutherAI/delphi/commit/0ceaa10bd7b1b7beb411c488ff348c06fb868a67))

* bump vllm dependency to latest

* change simulation scoring default to all at once for local models

* use Role class instead of hardcoding

* delete old oai simulator scorer

* refactor

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

* linter fix

* replace lambda with function

---------

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.1.0 (2025-06-17)

### Features

- Replaces print with logging ([#136](https://github.com/EleutherAI/delphi/pull/136),
  [`431d654`](https://github.com/EleutherAI/delphi/commit/431d6546c3a3a4ede107d5d4ebe0ffd75e5c923b))

* feat: replaces print with logging

* Change some infos to warnings in constructor

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Goncalo Paulo <30472805+SrGonao@users.noreply.github.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.0.2 (2025-06-02)

### Bug Fixes

- Initial release
  ([`6efa228`](https://github.com/EleutherAI/delphi/commit/6efa228c8d9c0fddc6cb21760c1adb1d51ffdc77))


## v0.0.1 (2025-04-21)

### Bug Fixes

- Add missing libraries
  ([`d6f7c72`](https://github.com/EleutherAI/delphi/commit/d6f7c72c0b9dd4fd12dc78aa75fd77d146b0199b))

- Add missing libraries
  ([`853d8ca`](https://github.com/EleutherAI/delphi/commit/853d8ca93256b8bda1395bab08199de45eb63926))

- Scope issue with sae causing it to not be loaded properly
  ([`94f40cc`](https://github.com/EleutherAI/delphi/commit/94f40cc8def426baa9be682e22c96d4c31a8b5ed))

- Scope issue with sae causing it to not be loaded properly
  ([`23caed7`](https://github.com/EleutherAI/delphi/commit/23caed746e0536da5a0739b6f0cdf12c678be467))

### Documentation

- Update README.md
  ([`4385b0b`](https://github.com/EleutherAI/delphi/commit/4385b0b3a9ee99fdbf5713a3f990ab5721b12d1e))

acessed -> accessed

- Update README.md
  ([`31b6896`](https://github.com/EleutherAI/delphi/commit/31b6896ae412903ab797da429f8c834a26a8dfed))

acessed -> accessed
