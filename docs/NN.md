# Neural Network Pipeline Documentation

This document explains the Neural Network (NN) architecture, data flow, and training processes used in the Claude Code Deobfuscator. The goal of the NN is to identify code chunks based on their structural "DNA" rather than their obfuscated names.

## Pipeline Architecture

The pipeline consists of three main stages:

1. **Bootstrapping**: Collecting "Gold Standard" ASTs from known libraries.
2. **Training**: Teaching a Triplet Network to recognize structural similarities.
3. **Inference (Vectorization)**: Generating fingerprints for unknown code chunks to align them with known logic.

---

## 1. Bootstrapping (`npm run bootstrap`)

Bootstrapping creates the ground truth data used for training. It installs specific versions of common libraries, bundles them with `esbuild`, and extracts their simplified structural ASTs.

- **Source Code**: `src/bootstrap_libs.js`
- **Logic**:
    - Selects libraries with known "version drift" (e.g., React 18 vs 19, Zod 3 vs 4).
    - Performs isolated installs in `ml/bootstrap_data/`.
    - Uses `src/analyze.js` to generate `simplified_asts.json` for each library.
    - Stores the final gold standards as `<lib_name>_v<version>_gold_asts.json` in `ml/bootstrap_data/`.

**Output Locations**: 

- `ml/bootstrap_data/` (Gold AST files)
- `cascade_graph_analysis/bootstrap/` (Temporary analysis artifacts)

---

## 2. Training (`npm run train`)

Training uses a Transformer Encoder architecture to learn an embedding space where similar code structures are close together and different ones are far apart.

- **Source Code**: `ml/train.py`
- **Network Architecture**: 
    - **TransformerCodeEncoder**: Transformer Encoder with learned positional embeddings and **CLS pooling**.
    - **CodeFingerprinter**: Two-channel Siamese output (structural + literal) with L2 normalization.
    - **Triplet Loss**: Optimizes (Anchor, Positive, Negative) triplets with vectorized hard-negative mining.

### Hyperparameter Sweep Analysis
Recent sweeps prioritize cross-library generalization and robust ranking metrics:

*   **Multi-library validation**: Validation splits can include multiple libraries to stress generalization.
*   **Same-library masking during eval**: When validating across multiple libraries, same-library negatives are masked to simulate real-world anchoring.
*   **MRR-first selection**: The sweep scores candidates by worst-case (min-library) MRR, with average MRR and margin as tie-breakers.

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `bootstrap_dir` | `./ml/bootstrap_data` | Directory containing the gold ASTs. |
| `--epochs` | `50` | Number of training iterations. |
| `--batch_size` | `64` | Number of triplets per optimization step. |
| `--lr` | `0.001` | Adam learning rate. |
| `--margin` | `0.5` | Triplet loss margin. |
| `--embed_dim` | `32` | AST node type embedding size. |
| `--hidden_dim` | `128` | Transformer feed-forward hidden size. |
| `--preset` | (unset) | Training preset (e.g., `production`). |
| `--lr_decay_epoch` | `0` | Apply learning-rate decay at this epoch (0 disables). |
| `--lr_decay_factor` | `1.0` | LR decay multiplier. |
| `--checkpoint_interval` | `0` | Save checkpoints every N epochs (0 disables). |
| `--checkpoint_dir` | (unset) | Override checkpoint directory. |
| `--finetune` | `false` | Load `ml/model.pth` (or latest checkpoint) to continue training. |
| `--force` | `false` | Force loading weights even if vocab size mismatches. |
| `--sweep` | `false` | Run hyperparameter sweep. |
| `--device` | `auto` | Device: `cuda`, `mps`, `cpu`, or `auto`. |
| `--val_library` | (unset) | Validation library name(s). Repeat or comma-separate. |
| `--val_lib_count` | `3` | Number of libraries to hold out when `--val_library` is not set. |
| `--val_split` | `0` | Use random split for validation (0 disables; overrides leave-library-out). |
| `--val_max_chunks` | `0` | Cap validation chunks (0 disables). |
| `--max_nodes` | `0` | Override context window size (0 = auto; can also use `ML_MAX_NODES`). |

---

## Model Technical Specifications

The system uses a custom **Multi-Channel Siamese Network** designed to process both topological and semantic signals.

### Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MAX_NODES` | `2048` | Default maximum AST sequence length (auto-tuned by device/weights at runtime). |
| `MAX_LITERALS` | `32` | Maximum number of hashed literals captured per chunk. |
| `Embedding Dim` | `32` | Dimension of the AST Node Type embeddings (default). |
| `Hidden Dim` | `128` | Dimension of the Transformer hidden state (default). |
| `Fingerprint Dim`| `64 x2` | Structural + literal L2-normalized output vectors. |
| `Learning Rate` | `0.001` | Default learning rate. |
| `Margin` | `0.5` | Default triplet loss margin. |

**Context Window Selection**
`MAX_NODES` is treated as a default ceiling. At runtime, the effective window is chosen in this order:
1. `--max_nodes` (CLI) or `ML_MAX_NODES` (env).
2. Checkpoint positional embedding length (if `ml/model.pth` exists).
3. Device heuristic: 2048 for large CUDA GPUs, 1024 for mid-tier, 512 for CPU, 256 for MPS.

### Architecture Detail

1.  **Structural Channel (Transformer)**:
    - Inputs: `(Batch, MAX_NODES)` token IDs.
    - Logic: Multi-Head Attention over flattened AST sequences with learned positional embeddings.
    - Pooling: CLS token aggregates the sequence.
    - Normalization: **L2 normalization** on the final 64-dim projection.
    - Padding: PAD tokens are masked during attention.
    - AST flattening skips noisy nodes like `EmptyStatement`, `DebuggerStatement`, and comment placeholders.
2.  **Literal Channel (Permutation-Invariant Pooling)**:
    - Inputs: Up to `MAX_LITERALS` hashed literal values; padding uses `-1.0`.
    - Encoding: Sin/Cos features from a 16-dim sinusoidal basis per literal hash.
    - Logic: Per-literal projection -> average pool -> 64-dim normalized vector.
    - Dropout guard: Literal kinds tagged as `const` or `path` are protected from dropout.

### Training Notes

*   **Hard negatives**: Within-batch mining masks isomorphic and same-library candidates when possible.
*   **Positive mixing**: ~10% positives are pulled from the same library to encourage library-family clustering.
*   **Structural noise**: 15% node-type masking, 20% junk node injection, and 30% sequence jittering on positives.
*   **Literal dropout**: 50% of non-protected literals are dropped (padding to `-1.0`).
*   **Loss weighting**: Structural and literal triplet losses are combined 0.7 / 0.3.
*   **Sweep batch size**: Sweeps use a fixed batch size of `64` inside `ml/train.py` regardless of CLI defaults.
*   **Early stopping**: Training stops after 10 stagnant epochs (min delta 0.001) on margin improvement.
*   **Evaluation metrics**: Margin and MRR are tracked; sweeps select on minimum library MRR.
*   **Leave-multi-library-out validation**: The training script holds out multiple libraries by default to improve validation diversity.
*   **Validation split override**: Use `--val_split` (or `ML_VAL_SPLIT`) to switch to a random split across all libraries.
*   **Preset: production**: Sets `margin=0.8`, `hidden_dim=256`, `embed_dim=32`, `lr=0.001`.

---

## Vocabulary & Specials

The model's vocabulary is dynamically generated from `@babel/types`.

### Structure
The vocabulary is built in a specific order:
- **Index 0 (`PAD`)**: Used for sequence padding.
- **Index 1 (`UNKNOWN`)**: Fallback for unrecognized node types.
- **Standard Babel Types**: The majority of the vocabulary consists of standard Babel AST node types (e.g., `Identifier`, `IfStatement`).
- **Custom "Built-in" Signals**: At the end of the vocabulary, there are custom signals detected during AST flattening, such as:
    - `Builtin_require`
    - `Builtin_defineProperty`
    - `Builtin_exports`
    - `Builtin_module`

The exact size of the vocabulary is not fixed. It will grow as new versions of Babel add more node types. To ensure the vocabulary used by the model is up-to-date, run `npm run sync-vocab`.

### Vocabulary Size
The vocabulary size is dynamic and synced with `@babel/types` via `npm run sync-vocab`. The model still initializes with `VOCAB_SIZE + 100` to provide a safety buffer for future Babel updates without immediately breaking model loading.

---

### Robust Model Loading

The training script only loads a checkpoint when `--finetune` is passed. When enabled, it:
- Automatically detects the vocabulary size of an existing `model.pth`.
- Warns if the current vocabulary (defined in `ml/constants.py`) differs.
- Partially loads weights and resizes both embedding and positional layers to allow resuming training despite vocabulary or context window changes.
- Exits with an error if `--finetune` is set but `ml/model.pth` is missing.
Vectorization uses the same partial-load approach and can infer `MAX_NODES` from the checkpoint's positional embeddings.

### Environment Overrides

Training supports several optional environment variables for defaults:

| Variable | Description |
| :--- | :--- |
| `ML_MAX_NODES` | Override context window size when `--max_nodes` is not set. |
| `ML_VAL_LIB_COUNT` | Default number of held-out libraries for validation. |
| `ML_VAL_SPLIT` | Default random split size (0 disables). |
| `ML_VAL_MAX_CHUNKS` | Default validation chunk cap (0 disables). |

Most training defaults can also be overridden with `ML_TRAIN_*` variables (see `ml/train.py`).

---

## 3. Inference / Vectorization (`ml/vectorize.py`)

During the analysis phase, the model generates **two 64-dimensional vectors** (structural + literal) for every chunk in the target codebase.

- **Source Code**: `ml/vectorize.py`
- **Invoked By**: `node run analyze` or `src/anchor_logic.js`.

### Arguments

| Argument | Description |
| :--- | :--- |
| `version_path` | Path to the version directory (e.g., `claude-analysis/v3.5`). |
| `--device` | Device: `cuda`, `mps`, `cpu`, or `auto`. |
| `--max_nodes` | Override context window size (0 = auto). |
| `--force` | Proceed without `ml/model.pth` (or despite vocab mismatch). Without it, vectorization fails if the model is missing. |

---

## Key Files & Locations

| Path | Description |
| :--- | :--- |
| `ml/constants.py` | Central registry of AST Node Types and model hyperparameters. |
| `ml/encoder.py` | PyTorch model definition (Transformer encoder + projection). |
| `ml/model.pth` | The trained weights. This is the "Brain" of the deobfuscator. |
| `ml/bootstrap_data/` | Ground truth data extracted from NPM libraries. |
| `src/sync_vocab.js` | Script to sync Babel types with `ml/constants.py`. |

## Model Card Reference

The current model card lives at `ml/model.md` and documents the most recent trained checkpoint (e.g., v1.3 "Pinky"). It may use non-default hyperparameters (larger context windows, higher hidden dims, or higher margins). Treat it as the authoritative record for that specific run, and use this document for code-level defaults.

## Vocabulary Health

The scripts perform a "Vocabulary Health Check" during execution. High percentages of `UNKNOWN` nodes indicate that the deobfuscator is encountering Babel types not present in `ml/constants.py`. In this case, run:

```bash
npm run sync-vocab
```

Then retraining the model (potentially with `--force`) is recommended.