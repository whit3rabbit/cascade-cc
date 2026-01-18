# Model Card: Cascade-CC Structural DNA Model (v1.0 "Lean Brain")

## 1. Model Summary

The **Cascade-CC Structural DNA Model** is a specialized Transformer-based Siamese Network designed to identify JavaScript code logic via its Abstract Syntax Tree (AST) topology. Unlike traditional code-search models that rely on variable names or comments, this model is "name-blind"—it learns the structural patterns (the "DNA") of logic to identify libraries even after aggressive minification and obfuscation.

## 2. Technical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Architecture** | Transformer Encoder | Siamese Triplet Configuration |
| **Model Size** | ~300 KB | Optimized weight footprint |
| **Embedding Dim** | 32 | Bottlenecked to force logic generalization |
| **Hidden Dim** | 128 | Capacity for complex branching/nesting |
| **Context Window** | 512 Nodes | Hardware-aware AST sequence length |
| **Output Vector** | 64-D | L2-Normalized Fingerprint |
| **Vocabulary** | 319+ Types | Synced with `@babel/types` (Jan 2026) |

## 3. Training Methodology

*   **Dataset:** 3,476 "Gold Standard" logic chunks from **87 modern NPM libraries** (React, Zod, AWS SDK, etc.). See Section 8 for the complete manifest.
*   **Loss Function:** Triplet Margin Loss ($Margin = 0.5$).
*   **Augmentation:** Synthetic structural mangling including Statement Shuffling, If-Statement swapping, and constant unfolding.
*   **Mining Strategy:** **Library-Aware Hard Negative Mining**. The model was forbidden from using negatives from the same library as the anchor, forcing it to learn cross-library logic discrimination.
*   **Validation:** Leave-One-Library-Out (LOLO) cross-validation.

## 4. Performance Metrics

Validated against complex recursive logic (e.g., `ajv`, `micromatch`) and shallow utilities.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Match Accuracy** | **100.00%** | Correct logic always closer than negatives. |
| **MRR (Mean Reciprocal Rank)** | **1.0000** | Correct logic is consistently the #1 match. |
| **Similarity Margin** | **1.0092** | Massive separation (Distance ~1.0 in normalized space). |
| **Inference Latency** | **< 2ms** | Optimized for real-time bundle anchoring. |

## 5. Intended Use

*   **Neural Anchoring:** Identifying known open-source libraries within the Claude Code minified bundle.
*   **Logic Alignment:** Transferring deobfuscation results from older versions of Claude to newer versions based on structural similarity.
*   **Founder Detection:** Identifying code with low similarity scores as proprietary "Founder" logic (target for LLM deobfuscation).

## 6. Limitations & Biases

*   **Small Chunks:** Logic blocks with fewer than 10 AST nodes may produce "collisions" (non-unique fingerprints).
*   **Language Specificity:** Currently tuned specifically for ECMAScript (JavaScript/TypeScript) AST node types.
*   **Pure Structure:** Without the **Literal Channel**, the model cannot distinguish between two functions with identical control flow that operate on different string constants.

## 7. Version History
*   **v1.0 (Current):** Switched to Embed 32/Hidden 128 "Lean Brain" architecture. Achieved highest recorded margin (1.009) with 50% reduction in model size. Improved generalization via library-aware masking.

---

### Recommended Environment for Inference:
*   **Device:** `cuda` (Recommended), `mps`, or `cpu`.
*   **Context:** `MAX_NODES=512`.
*   **Threshold:** A similarity score of **> 0.95** should be treated as a High-Confidence identity match.

## 8. Training Library Manifest (Version Drift Aware)

The model was trained against multiple versions of the following libraries to ensure robustness against "Version Drift."

| Library Group | Included Libraries & Versions |
| :--- | :--- |
| **Core UI / React** | `react` (18.3.1, 19.2.3, latest), `ink` (6.6.0, latest), `@inkjs/ui` (2.0.0, latest) |
| **Schema & Logic** | `zod` (3.23.8, 4.2.1, 4.3.5, latest), `ajv` (8.17.1, latest), `lodash-es` (4.17.21, 4.17.22, latest) |
| **AI & Cloud SDKs** | `@anthropic-ai/sdk` (0.40.0, 0.71.2, latest), `@anthropic-ai/bedrock-sdk` (0.26.0, latest), `@anthropic-ai/vertex-sdk` (0.14.0, latest), `@aws-sdk/client-bedrock` (3.962.0), `@aws-sdk/client-bedrock-runtime` (3.962.0), `@aws-sdk/client-s3` (3.958.0), `@aws-sdk/client-sts` (3.958.0), `@aws-sdk/credential-providers` (3.958.0) |
| **Parsing & AST** | `tree-sitter` (0.21.0, latest), `tree-sitter-typescript` (0.23.2, latest), `web-tree-sitter` (0.26.3, latest), `js-yaml` (4.1.1, latest), `parse5` (latest), `domino` (latest) |
| **Networking** | `axios` (0.27.2, 1.13.2, 1.7.9, latest), `ws` (8.18.3, latest), `https-proxy-agent` (7.0.6, latest) |
| **Observability** | `@opentelemetry/api` (1.9.0, latest), `@opentelemetry/core` (2.2.0, latest), `@opentelemetry/sdk-trace-node` (2.2.0, latest), `@sentry/node` (10.32.1, latest), `@segment/analytics-node` (latest) |
| **CLI & Terminal** | `commander` (14.0.2, latest), `chalk` (5.6.2, latest), `execa` (9.6.1, latest), `prompts` (latest), `ink-link` (5.0.0), `ink-select-input` (6.2.0), `ink-spinner` (5.0.0), `ink-text-input` (6.0.0), `cli-highlight` (2.1.11), `cli-table3` (0.6.5) |
| **Utilities** | `date-fns` (3.6.0, latest), `uuid` (9.0.0, latest), `semver` (7.7.3, latest), `lru-cache` (11.2.4, latest), `memoize` (10.2.0, latest), `micromatch` (4.0.8, latest), `mime-types` (3.0.2, latest), `chokidar` (5.0.0, latest), `proper-lockfile` (4.1.2, latest), `open` (11.0.0, latest) |
| **Media & Content** | `sharp` (0.34.5, latest), `@resvg/resvg-js` (2.6.2, latest), `pdf-parse` (2.4.5, latest), `turndown` (7.2.2, latest), `marked` (17.0.1, latest), `gray-matter` (4.0.3, latest) |
| **System** | `is-unicode-supported` (2.1.0), `string-width` (8.1.0), `supports-hyperlinks` (4.4.0), `wcwidth` (1.0.1), `word-wrap` (1.2.5), `wrap-ansi` (9.0.2), `shell-quote` (1.8.3), `diff` (8.0.2), `ansi-escapes` (7.2.0), `ansi-styles` (6.2.3) |
| **Miscellaneous** | `@modelcontextprotocol/sdk` (1.25.1, latest), `jsdom` (27.4.0, latest), `fuse.js` (7.1.0, latest), `grapheme-splitter` (1.0.4, latest), `highlight.js` (11.11.1, latest), `html-entities` (latest), `localforage` (latest), `ordered-map` (0.1.0), `fflate` (latest), `figures` (6.1.0), `abort-controller` (3.0.0), `statsig-js` (5.1.0), `yoga-layout-prebuilt` (1.10.0), `xmlbuilder2` (3.1.1), `xss` (latest), `tslib` (latest), `uri-js` (latest) |