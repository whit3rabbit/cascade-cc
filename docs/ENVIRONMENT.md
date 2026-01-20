# Environment Configuration Guide

This document explains the environment variables used by the Claude Code Cascade Analyzer. These variables are defined in the `.env` file (copied from `.env.example`).

## LLM Configuration

| Variable | Description |
| :--- | :--- |
| `LLM_PROVIDER` | The LLM API provider. Options are `gemini` or `openrouter`. |
| `LLM_MODEL` | The specific model identifier (e.g., `gemini-2.0-flash` for Gemini or `google/gemini-2.0-flash-exp:free` for OpenRouter). |
| `GEMINI_API_KEY` | Your Google AI Studio / Gemini API key. |
| `OPENROUTER_API_KEY` | Your OpenRouter API key. |

## Detection & Analysis Parameters

These variables tune the static analysis and anchoring logic.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ANCHOR_KEY_CONFIDENCE` | `0.9` | The minimum confidence score required to accept a structural key match. |
| `ANCHOR_NAME_CONFIDENCE`| `0.85` | The minimum confidence score required to accept a name-based match. |
| `ANCHOR_SIMILARITY_THRESHOLD` | `0.80` | The minimum vector similarity (0.0 to 1.0) needed for a chunk to be considered a match. |
| `LIBRARY_MATCH_THRESHOLD` | `0.95` | Similarity required to auto-label a chunk as a vendor library match. |
| `ANCHOR_LOCK_THRESHOLD` | `0.98` | Similarity required to lock a match and apply names with high confidence. |
| `ANCHOR_LOCK_CONFIDENCE` | `0.99` | Confidence score assigned to locked matches. |
| `MARKOV_DAMPING_FACTOR`| `0.85` | Used in the PageRank-style Centrality algorithm. Higher values increase the influence of long-range connections. |
| `CHUNKING_TOKEN_THRESHOLD` | `2000` | The target number of tokens per code chunk during analysis. |
| `SPREADING_THRESHOLD_RATIO` | `0.3` | Threshold for property spreading activation. |
| `SPREADING_THRESHOLD_COUNT` | `2` | Minimum number of hits for property name propagation. |
| `ML_LITERAL_DROPOUT` | `0.5` | Training regularization: probability of masking literal features. |
| `ML_NODE_MASKING` | `0.15` | Training regularization: probability of masking node types. |
| `ML_SEQ_JITTER` | `0.3` | Training regularization: probability of sequence jittering. |

## Role Classification Thresholds

The system uses Graph Theory metrics to identify the role of each chunk (e.g., Vendor vs. Core).

| Variable | Description |
| :--- | :--- |
| `VENDOR_LIBRARY_IN_DEGREE` | In-degree threshold to classify a chunk as a Vendor library. |
| `VENDOR_LIBRARY_OUT_DEGREE` | Out-degree threshold to classify a chunk as a Vendor library. |
| `CORE_ORCHESTRATOR_IN_DEGREE` | In-degree threshold to classify a chunk as a Core orchestrator. |
| `CORE_ORCHESTRATOR_OUT_DEGREE` | Out-degree threshold to classify a chunk as a Core orchestrator. |

These thresholds are actively used by `src/classify_logic.js` to separate utility libraries from application logic.

---

> [!TIP]
> If you are experiencing high rates of "unknown" nodes or low-confidence matches, consider lowering the `ANCHOR_SIMILARITY_THRESHOLD` slightly or re-running `npm run sync-vocab` followed by a retraining session.
