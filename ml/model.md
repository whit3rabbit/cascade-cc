\# Model Card: Cascade-CC Structural DNA Model (v1.2 "High-Res Brain")

## 1. Model Summary

The **Cascade-CC Structural DNA Model** is a specialized Transformer-based Siamese Network designed to identify JavaScript code logic via its Abstract Syntax Tree (AST) topology. Version 1.2 features an expanded hidden dimension and a doubled context window, allowing it to capture the "fingerprints" of complex vendor libraries (e.g., AWS SDK, Sentry, Anthropic SDK) with high precision, even after aggressive minification.

The "Brain" scales its resolution (context window) based on the hardware it is running on.

## 2. Technical Specifications

*   **Architecture:** Siamese Transformer Encoder
*   **Context Window:** 2048 AST Nodes (Auto-scales based on VRAM)
*   **Embedding Dimension:** 32
*   **Hidden Dimension:** 256
*   **Peak Performance (v1.2):** Epoch 8
*   **Final AvgLib MRR:** 0.9691
*   **Final MinLib MRR:** 0.8963
*   **Validation Match Accuracy:** 91.76%
*   **Similarity Spread (Margin):** 0.4031 (Spread between Pos/Neg distances)

## 3. Training Methodology

*   **Dataset:** 4,320 "Gold Standard" logic chunks from **110+ modern NPM libraries**.
*   **Loss Function:** Triplet Margin Loss ($Margin = 1.2$).
*   **Augmentation:** Nuclear-grade structural mangling: 50% Literal Dropout, 15% Node Masking, and 30% Sequence Jittering.
*   **Mining Strategy:** **Vectorized Hard Negative Mining**. The model identifies the "hardest" negatives (most structurally similar non-matches) within each batch to maximize discrimination.
*   **Validation Strategy:** Leave-Multi-Library-Out (LMLO). Validated against 10 completely unseen libraries including `aws-sdk`, `anthropic-ai/sdk`, and `sentry/node`.
*   **Training Command:** `node run train --device cuda --batch_size 64 --hidden_dim 256 --max_nodes 2048 --margin 1.2 --lr 0.00002 --finetune`

## 4. Performance Metrics (v1.2 Final Run)

Validated against a high-stress pool of 255 unseen structural chunks.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Match Accuracy** | **91.76%** | Correct logic is closer than the hardest negative in 9/10 cases. |
| **Avg-Library MRR** | **0.9691** | On average, the correct match is ranked #1.03. |
| **Min-Library MRR** | **0.8963** | Even for the hardest library, the match is usually #1 or #2. |
| **Inference Latency** | **~4ms** | Slightly higher than v1.0 due to 256-dim hidden layer. |

**Latest Run Snapshot (High-Resolution Tuning):**
- **Epoch 6:** Accuracy 90.98%, AvgLib MRR 0.9139, MinLib MRR 0.7180
- **Epoch 8 (PEAK):** Accuracy 91.76%, AvgLib MRR 0.9691, MinLib MRR 0.8963
- **Epoch 9:** Accuracy 87.06%, AvgLib MRR 0.8966 (Early stopping triggered due to positive cluster drift).

## 5. Intended Use

*   **Neural Anchoring:** Identifying known open-source libraries within the Claude Code minified bundle.
*   **Vendor Lock-In:** Automatically tagging logic as `vendor` when similarity exceeds **0.95**.
*   **Logic Alignment:** Transferring deobfuscation results across version drifts (e.g., mapping `axios@1.7.8` logic to `axios@1.7.9`).

## 6. Limitations & Biases

*   **Context Saturation:** At 2048 nodes, very small helper functions (e.g., `isObject`) may appear identical across different libraries.
*   **Literal Blindness:** Highly effective against obfuscation, but requires the **Literal Channel** to distinguish between functions with identical control flow (e.g., two different string-concatenation utilities).

## 7. Version History
*   **v1.0 (Lean):** Embed 32/Hidden 128. Fastest inference, good for simple logic.
*   **v1.1 (Production):** Margin 0.8, LR 0.001. Balanced performance.
*   **v1.2 (High-Res):** **Current.** Embed 32/Hidden 256, 2048 Nodes. Significant jump in `MinLib MRR` (0.50 -> 0.89). Optimized for deep identification of complex SDKs (AWS, Anthropic, Sentry).

---

### Recommended Environment for Inference:
*   **Device:** `cuda` (Highly Recommended) or `mps`.
*   **Thresholds:** 
    *   **> 0.98:** **LOCKED.** Apply automatic symbol renaming with 99% confidence.
    *   **> 0.95:** **GOLDEN.** High-confidence library identification.
    *   **> 0.80:** **CANDIDATE.** Potential match; requires LLM verification.

## 8. Training Library Manifest (v1.2 Update)
*(Manifest remains consistent with previous version, but now includes deep structural fingerprints for the following high-complexity groups)*

| Library Group | Key Targets Successfully Fingerprinted |
| :--- | :--- |
| **AI SDKs** | `@anthropic-ai/sdk`, `@anthropic-ai/bedrock-sdk`, `@anthropic-ai/vertex-sdk` |
| **Cloud Infrastucture** | `@aws-sdk/client-s3`, `@aws-sdk/client-bedrock-runtime`, `@aws-sdk/client-sts` |
| **Observability** | `@sentry/node`, `@opentelemetry/api`, `@segment/analytics-node` |
| **Complex Parsing** | `ajv`, `zod`, `marked`, `tree-sitter-typescript` |