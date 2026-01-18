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
    - **CodeStructureEncoder**: Uses a Transformer Encoder with learnable positional embeddings.
    - **Literal Channel**: A separate pathway that hashes literals (Strings/Numbers) to preserve semantic signals.
    - **Triplet Loss**: Optimizes the model using (Anchor, Positive, Negative) triplets with Hard Negative Mining.

### Hyperparameter Sweep Analysis
Based on extensive sweeps, the model has been tuned for the following "Brain" configuration:

*   **Efficiency > Complexity**: The smallest architecture (`Embed: 32`, `Hidden: 64`) achieves **100% accuracy** on synthetic structural DNA.
*   **Robustness**: The Transformer architecture effectively ignores "noise" (mangled names, dead code) and focuses on AST topology.
*   **Convergence**: A learning rate of **0.001** provides the fastest stable convergence.

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `bootstrap_dir` | `./ml/bootstrap_data` | Directory containing the gold ASTs. |
| `--epochs` | `10` | Number of training iterations. |
| `--batch_size` | `32` | Number of triplets per optimization step. |
| `--force` | `False` | Force loading a model even if the vocabulary size mismatches. |

---

## Model Technical Specifications

The system uses a custom **Multi-Channel Siamese Network** designed to process both topological and semantic signals.

### Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MAX_NODES` | `512` | Maximum length of the AST sequence. Optimized for Transformers. |
| `MAX_LITERALS` | `32` | Maximum number of hashed literals captured per chunk. |
| `Embedding Dim` | `32` | Dimension of the AST Node Type embeddings (Optimized). |
| `Hidden Dim` | `64` | Dimension of the Transformer hidden state. |
| `Fingerprint Dim`| `128` | Final L2-normalized output vector size. |
| `Learning Rate` | `0.001` | Optimal learning rate for fast convergence. |
| `Margin` | `0.2` | Optimal triplet loss margin. |

### Architecture Detail

1.  **Structural Channel (Transformer)**:
    - Inputs: `(Batch, 512)` token IDs.
    - Logic: Uses Multi-Head Attention to capture global structural dependencies.
    - Normalization: Applies **L2 Normalization** to the final embedding.

---

## Vocabulary & Specials

The model's vocabulary is dynamically generated from `@babel/types`.

### Structure
- **Index 0 (`PAD`)**: Used for sequence padding.
- **Index 1 (`UNKNOWN`)**: Fallback for unrecognized node types.
- **Indices 2-314**: Standard Babel node types (e.g., `Identifier`, `IfStatement`).
- **Indices 315-318**: Custom "Built-in" signals detected during flattening:
    - `Builtin_require`
    - `Builtin_defineProperty`
    - `Builtin_exports`
    - `Builtin_module`

### Vocabulary Size
The default total vocabulary size is **319**. However, the model is initialized with `VOCAB_SIZE + 5` to provide a safety buffer for future Babel updates without immediately breaking model loading.

---

### Robust Model Loading

The training script includes a robust loading mechanism that:
- Automatically detects the vocabulary size of an existing `model.pth`.
- Warns if the current vocabulary (defined in `ml/constants.py`) differs.
- If `--force` is used, it partially loads weights and resizes the embedding layer to allow resuming training despite vocabulary changes.

---

## 3. Inference / Vectorization (`ml/vectorize.py`)

During the analysis phase, the model is used to generate 64-dimensional vectors (fingerprints) for every chunk in the target codebase.

- **Source Code**: `ml/vectorize.py`
- **Invoked By**: `node run analyze` or `src/anchor_logic.js`.

### Arguments

| Argument | Description |
| :--- | :--- |
| `version_path` | Path to the version directory (e.g., `claude-analysis/v3.5`). |
| `--force` | Force loading the model despite vocabulary mismatches. |

---

## Key Files & Locations

| Path | Description |
| :--- | :--- |
| `ml/constants.py` | Central registry of AST Node Types and model hyperparameters. |
| `ml/encoder.py` | PyTorch model definition (LSTM + Linear layers). |
| `ml/model.pth` | The trained weights. This is the "Brain" of the deobfuscator. |
| `ml/bootstrap_data/` | Ground truth data extracted from NPM libraries. |
| `src/sync_vocab.js` | Script to sync Babel types with `ml/constants.py`. |

## Vocabulary Health

The scripts perform a "Vocabulary Health Check" during execution. High percentages of `UNKNOWN` nodes indicate that the deobfuscator is encountering Babel types not present in `ml/constants.py`. In this case, run:

```bash
npm run sync-vocab
```

Then retraining the model (potentially with `--force`) is recommended.
