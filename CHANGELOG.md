# Changelog

## v0.1.3 (2026-03-12)

- Make `log_prob` an explicit classifier constructor option
  ([fb3aa86](https://github.com/EleutherAI/delphi/commit/fb3aa863487683285a04f39f3c3134aa6cddb956)).
- Add uv quality gates and test discovery configuration
  ([f851189](https://github.com/EleutherAI/delphi/commit/f851189d1f61b2886176e37babc84c3198eb5a32)).

## v0.1.2 (2026-03-12)

- Add configurable `max_memory` for the offline explainer and update vLLM
  prompt formatting and the minimum supported vLLM version
  ([#156](https://github.com/EleutherAI/delphi/pull/156)).
- Fix multiple false-positive highlighting and intruder example formatting,
  and simplify pipeline error reporting
  ([#153](https://github.com/EleutherAI/delphi/pull/153)).

## v0.1.1 (2025-08-27)

- Restore simulator compatibility with vLLM, default local simulation to the
  all-at-once mode, and refactor simulator prompt handling
  ([0ceaa10](https://github.com/EleutherAI/delphi/commit/0ceaa10bd7b1b7beb411c488ff348c06fb868a67)).

## v0.1.0 (2025-06-17)

- Replace print statements with logging
  ([#136](https://github.com/EleutherAI/delphi/pull/136)).

## v0.0.2 (2025-06-02)

This release records an empty commit titled "initial release"
([6efa228](https://github.com/EleutherAI/delphi/commit/6efa228c8d9c0fddc6cb21760c1adb1d51ffdc77)).
It is a release marker, not the start of the project's history; v0.0.1 was
already published. The marker itself contains no code changes.

## v0.0.1 (2025-04-21)

- Add missing dependencies
  ([d6f7c72](https://github.com/EleutherAI/delphi/commit/d6f7c72c0b9dd4fd12dc78aa75fd77d146b0199b),
  [853d8ca](https://github.com/EleutherAI/delphi/commit/853d8ca93256b8bda1395bab08199de45eb63926)).
- Fix an SAE scope issue that prevented loading
  ([94f40cc](https://github.com/EleutherAI/delphi/commit/94f40cc8def426baa9be682e22c96d4c31a8b5ed),
  [23caed7](https://github.com/EleutherAI/delphi/commit/23caed746e0536da5a0739b6f0cdf12c678be467)).
- Correct a README typo
  ([4385b0b](https://github.com/EleutherAI/delphi/commit/4385b0b3a9ee99fdbf5713a3f990ab5721b12d1e),
  [31b6896](https://github.com/EleutherAI/delphi/commit/31b6896ae412903ab797da429f8c834a26a8dfed)).
