# Model Card: Cascade-CC Structural DNA Model (v1.4 "Pinky Hardened")

## 1. Model Summary

The **Cascade-CC Structural DNA Model (v1.4)** represents the "Hardened" iteration of our Siamese Transformer. By doubling the hidden dimension to **512** and increasing the triplet margin to **1.5**, this version has learned to discriminate between extremely similar code patterns (like different JSON validators or URI parsers) that previous versions found ambiguous. 

## 2. Technical Specifications

*   **Architecture:** Siamese Transformer Encoder with CLS Pooling.
*   **Context Window:** 1024 AST Nodes.
*   **Embedding Dimension:** 128 (Node Type IDs).
*   **Hidden Dimension:** 512 (High-resolution logic discrimination).
*   **Peak Performance:** Epoch 51 (Hardening Phase).
*   **Final Avg-Library MRR:** 0.8086
*   **Final Match Accuracy:** 84.81% (Generalization across 12 unseen libraries).
*   **Similarity Spread (Margin):** 0.1816 (0.2437 Neg / 0.0621 Pos).

## 3. Training Methodology

*   **Dataset:** ~15,500 "Gold Standard" logic chunks from high-complexity NPM libraries.
*   **Loss Function:** Hardened Triplet Margin Loss ($Margin = 1.5$).
*   **Augmentation:** 50% Literal Dropout, 15% Node Masking, 30% Sequence Jittering.
*   **Optimization:** Adam Optimizer ($LR = 0.00002$) for fine-tuning stability.
*   **Validation Strategy:** Leave-Multi-Library-Out (LMLO). Validated against 12 completely unseen libraries including `ajv`, `cli-highlight`, `fuse.js`, and `axios`.
*   **Training Command:** 
    `node run train --finetune --device cuda --batch_size 32 --epochs 30 --lr 0.00002 --margin 1.5 --embed_dim 128 --hidden_dim 512 --val_max_chunks 500 --val_lib_count 12`

## 4. Performance Metrics (v1.4 Hardened - Epoch 51)

Validated against a high-stress pool of 500 unseen structural chunks.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Match Accuracy** | **84.81%** | Identifies the correct logic family on the first try in ~8.5/10 cases. |
| **MRR (Ranking)** | **0.8086** | The correct match is ranked #1.2 on average. |
| **Similarity Spread**| **0.1816** | Aggressive separation; extremely low chance of "False Positive" library matches. |

### Hardening Analysis
The move to a **1.5 Margin** forced the model to ignore superficial similarities. While the raw accuracy number is lower than a "soft" model (0.5 margin), the **reliability** of the matches is significantly higher. Matches with similarity $>0.90$ are now almost certainly identical logic.

## 5. Intended Use

*   **Hardened Anchoring:** Use as the primary engine for `npm run anchor`.
*   **Library Collapsing:** Automatically identifies and labels standard NPM modules within the Claude bundle, allowing the LLM to skip them and save tokens.
*   **Symbol Persistence:** Identifying renamed logic across Claude versions 2.1.x with 95%+ confidence.

## 6. Limitations

*   **Data-Only Chunks:** Still struggles with "Flat Data" libraries (e.g., `figures`, `wcwidth`) which lack control flow.
*   **VRAM Intensive:** Requires 15GB VRAM for 1024-node context at batch size 32.

## 7. Version History
*   **v1.2 (High-Res):** Embed 32 / Hidden 256. Good general performance.
*   **v1.3 (Pinky):** Initial 128/512 attempt. Hit OOM issues.
*   **v1.4 (Hardened):** **Current.** 128/512 architecture with 1.5 Margin. Optimized for zero-collision anchoring.

---

### Operational Thresholds (Recommended):
*   **> 0.96:** **LOCKED.** Apply automatic symbol renaming without LLM verification.
*   **> 0.88:** **GOLDEN.** Highly likely match; LLM should prioritize these names.
*   **> 0.75:** **CANDIDATE.** Structural similarity detected; requires semantic context.