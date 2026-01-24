# Model Card: Cascade-CC Structural DNA Model (v1.3 "Pinky")

## 1. Model Summary

The **Cascade-CC Structural DNA Model (v1.3)** is a specialized Transformer-based Siamese Network designed to identify JavaScript code logic via its Abstract Syntax Tree (AST) topology. This version is optimized for "Deep Anchoring," capturing the fingerprints of complex libraries (e.g., AWS SDK, Zod, React) with high precision, even when obfuscated. The model treats code as a structural sequence, prioritizing the "DNA" of the logic over variable names or literal values.

## 2. Technical Specifications

*   **Architecture:** Siamese Transformer Encoder with CLS Pooling.
*   **Context Window:** 1024 AST Nodes (Optimized for CUDA hardware).
*   **Embedding Dimension:** 32 (Node Type IDs).
*   **Hidden Dimension:** 256 (Expanded for higher logic discrimination).
*   **Peak Performance:** Epoch 38 (Converged).
*   **Final Avg-Library MRR:** 0.9545
*   **Final Min-Library MRR:** 0.8445
*   **Validation Match Accuracy:** 92.07%
*   **Similarity Spread (Margin):** 0.1741 (0.2636 Neg / 0.0895 Pos).

## 3. Training Methodology

*   **Dataset:** 7,559 "Gold Standard" logic chunks extracted from modern NPM libraries including high-complexity SDKs.
*   **Loss Function:** Triplet Margin Loss ($Margin = 0.8$).
*   **Augmentation:** Nuclear-grade structural mangling: 50% Literal Dropout, 15% Node Masking, and 30% Sequence Jittering (includes random junk injection and commutative swapping).
*   **Optimization:** Adam Optimizer ($LR = 0.00005$) with a 50% decay at Epoch 20.
*   **Mining Strategy:** **Vectorized Hard Negative Mining**. The model dynamically identifies the most structurally similar non-matches within each batch to maximize discrimination.
*   **Validation Strategy:** Leave-Multi-Library-Out (LMLO). Validated against 10 completely unseen libraries including `aws-sdk`, `sharp`, `zod`, and `lru-cache`.
*   **Training Command:** 
    `node run train --device cuda --batch_size 64 --epochs 50 --lr 0.00005 --margin 0.8 --embed_dim 32 --hidden_dim 256 --lr_decay_epoch 20 --lr_decay_factor 0.5 --val_max_chunks 300 --val_lib_count 10`

## 4. Performance Metrics (v1.2 Final Run - Epoch 38)

Validated against a high-stress pool of 227 unseen structural chunks across diverse libraries.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Match Accuracy** | **92.07%** | Correct logic is closer than the hardest negative in >9/10 cases. |
| **Avg-Library MRR** | **0.9545** | On average, the correct match is the #1 recommendation. |
| **Min-Library MRR** | **0.8445** | Even for the hardest library (`cli-highlight`), the match is ranked #1 or #2. |
| **Similarity Spread**| **0.1741** | Healthy separation between positive and negative logic clusters. |

### Recovery Analysis
This model demonstrated exceptional learning resilience. Mid-training (Epoch 23), the `lru-cache` library was nearly invisible to the model (MRR 0.07). By the final epoch, the model successfully resolved these complex patterns, bringing all validation libraries above the **0.84 MRR** threshold.

## 5. Intended Use

*   **Neural Anchoring:** Identifying known open-source libraries within the Claude Code minified bundle.
*   **Vendor Lock-In:** Automatically tagging logic as `vendor` when similarity exceeds **0.95**.
*   **Proprietary Mapping:** Recognizing proprietary "Founder" code across versions by identifying logic re-used from previous manual deobfuscation efforts.

## 6. Limitations & Biases

*   **Junk Sensitivity:** While robust against masking, extremely high levels of dead-code injection (>40%) may degrade accuracy.
*   **Utility Collisions:** Extremely small helper functions (e.g., `isObject` or `noop`) may appear identical across different libraries due to their simple AST structure.
*   **Literal Dependency:** While the model is structural, it utilizes a 30% weighted "Literal Channel" to distinguish between functions with identical control flow but different constant data.

## 7. Version History
*   **v1.0 (Lean):** Embed 32 / Hidden 128. Fast inference, designed for simple script deobfuscation.
*   **v1.1 (Production):** Margin 0.8, LR 0.001. Balanced performance for general NPM libraries.
*   **v1.2 (High-Res):** **Current.** Embed 32 / Hidden 256, 1024-2048 Nodes. Specifically tuned for complex SDK discrimination. Successfully bridges the "MRR Gap" for libraries with highly abstracted logic.

---

### Recommended Environment for Inference:
*   **Device:** `cuda` (Highly Recommended) or `mps`.
*   **Inference Latency:** ~6ms per chunk.
*   **Operational Thresholds:** 
    *   **> 0.98:** **LOCKED.** Apply automatic symbol renaming with 99% confidence.
    *   **> 0.95:** **GOLDEN.** High-confidence library identification.
    *   **> 0.85:** **CANDIDATE.** Potential match; recommended for LLM-verified renaming.

## 8. Training Library Manifest (v1.2 Key Targets)

| Library Group | Key Targets Successfully Fingerprinted |
| :--- | :--- |
| **AI & Cloud** | `@aws-sdk/client-bedrock-runtime`, `@anthropic-ai/sdk`, `statsig-js` |
| **Core Utilities** | `lru-cache`, `zod`, `cli-highlight`, `sharp`, `word-wrap` |
| **UI & Terminal** | `ink`, `@inkjs/ui`, `is-unicode-supported`, `chalk` |
| **Parser Logic** | `ajv`, `marked`, `tree-sitter-typescript` |