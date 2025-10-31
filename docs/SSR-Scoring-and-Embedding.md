# Hybrid Anchored Scoring & Embedding Fallback

## Overview
The scoring pipeline now converts qualitative Semantic Similarity Rating (SSR) analyses
into stable numeric outputs using the Hybrid Anchored Scoring (HAS) algorithm. HAS
combines discrete anchor labels from the SSR text with distribution-aware embedding
similarity so that semantically close concepts remain well separated numerically. The
scheme powers both the spreadsheet scoring worker and interview transcript enrichment.

## Anchor weights
The SSR assistant must begin each concept line with one of the canonical anchors. HAS
maps anchors to base weights before applying embedding modifiers:

| Anchor      | Weight |
|-------------|--------|
| Core        | 0.95   |
| Strong      | 0.80   |
| Reasonable  | 0.55   |
| Weak        | 0.25   |
| None        | 0.05   |

Anchors default to **Weak** if the SSR text omits a label. Guarantees: Core scores are
clamped to ≥ 0.80 and None scores to ≤ 0.20 (configurable via environment overrides).

## Text normalisation
Before embedding, all utterances and concept definitions are normalised with Unicode
NFKC, variation selectors are removed, and whitespace is collapsed. Optional emoji-to-
text substitution can be enabled with `ANYAI_EMBED_MAP_EMOJI=1`.

## Embedding similarity
For each utterance *U* and concept *Cᵢ*, we compute cosine similarity between
`embed(U)` and `embed(Cᵢ)`. The raw similarities are min–max normalised to `rᵢ`. To
widen the mid-range, the values are passed through a logistic amplifier:

```
pᵢ = sigmoid(β * (rᵢ - 0.5)),  with β defaulting to 3.0
```

## Hybrid Anchored Score (HAS)
The final absolute score for concept *i* is

```
finalᵢ = λ * w(anchorᵢ) + (1 - λ) * pᵢ
```

where λ defaults to 0.7. The pipeline also exposes
`relative_rank_scoreᵢ = rᵢ` for visualisations.

### Example (Reasonable vs Weak)
Given embeddings that yield similarities `[0.62, 0.59]` and anchors `Reasonable` vs
`Weak`, HAS produces a gap ≥ 0.25 by default:

* `Reasonable`: final ≈ 0.63
* `Weak`: final ≈ 0.38
* Separation: ≥ 0.25 (configurable via β/λ if needed)

## SSR vs numeric outputs
* SSR LLM responses must remain natural-language rationales: `{"analyses": [...]}`.
* Numeric scoring is computed **only** inside HAS and surfaced via
  `ScoreResult.absolute_scores` (single-number compatibility uses the same values).
* Interview transcripts can optionally attach `concepts` and `analyses`; HAS populates
  `transcript["scoring"]` so downstream tools share the same numeric semantics.

## Interview reuse
`InterviewJobManager._apply_hybrid_scores` inspects transcripts for `concepts` and
anchor-labelled `analyses`. When present, it runs HAS on the combined persona answer
text and appends absolute/relative scores plus anchor metadata under
`transcript["scoring"]`. The routine logs the same structured events as the main
pipeline for observability.

## Concurrency rule
When SSR mode is enabled we encourage small batch sizes but no longer force them:

* UI は SSR ON/OFF を問わずカテゴリ同梱数を編集可能に保つ。
* Job API は送信された `batch_size` を尊重し、必要な検証のみを行う。
* Worker 起動時は `max_category_cols` を `batch_size` 以上に正規化するだけで強制固定はしない。

## Embedding fallback
`embed_with_fallback` now wraps the primary provider (default: OpenAI
`text-embedding-3-small`) and automatically fails over to
`gemini-embedding-001` when the primary raises, times out, or returns empty vectors.
Structured log events:

* `evt=embed_request` / `evt=embed_response`
* `evt=embed_fallback primary=<model> fallback=gemini-embedding-001`
* `evt=embedding_model_selected provider=<provider>`

If both providers fail the error bubbles to the caller.

## Logging taxonomy
All scoring stages emit grep-friendly single-line events at DEBUG level:

* Pipeline lifecycle: `evt=job_start`, `evt=job_done`, `evt=job_fail`
* SSR toggles & guards: `evt=ssr_on`, `evt=ssr_off`, `evt=ssr_concurrency_forced`
* Embedding lifecycle: `evt=embed_request`, `evt=embed_response`,
  `evt=embed_fallback`, `evt=embedding_model_selected`
* Scoring signals: `evt=anchor_parsed`, `evt=score_components`
* Cache: `evt=cache_hit`, `evt=cache_miss`, `evt=cache_store`
* Output sinks: `evt=write_outcome`

Raw utterances are never logged unless `ANYAI_LOG_ALLOW_TEXT=1` is set.

## Configuration knobs
* `ANYAI_HAS_LAMBDA` – adjusts anchor dominance (default 0.7)
* `ANYAI_HAS_BETA` – adjusts logistic amplification (default 3.0)
* `ANYAI_HAS_CORE_MIN` / `ANYAI_HAS_NONE_MAX` – guarantee bounds for extreme anchors
* `ANYAI_HAS_DEFAULT_ANCHOR` – fallback anchor when parsing fails (default Weak)
* `ANYAI_EMBED_MAP_EMOJI` – enables emoji → text name normalisation
* `ANYAI_LOG_LEVEL` – controls global log level (DEBUG by default)

These settings apply uniformly to the spreadsheet worker, interview scoring, and any
future consumers of the HAS functions.
