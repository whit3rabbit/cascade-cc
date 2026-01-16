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

Training uses a Siamese Triplet Network to learn an embedding space where similar code structures are close together and different ones are far apart.

- **Source Code**: `ml/train.py`
- **Network Architecture**: 
    - **CodeStructureEncoder**: Uses an Embedding layer followed by a Bidirectional LSTM.
    - **Literal Channel**: A separate pathway that hashes literals (Strings/Numbers) to preserve semantic signals while ignoring specific values.
    - **Triplet Loss**: Optimizes the model using (Anchor, Positive, Negative) triplets.

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
| `MAX_NODES` | `2048` | Maximum length of the AST sequence. Longer chunks are truncated. |
| `MAX_LITERALS` | `32` | Maximum number of hashed literals captured per chunk. |
| `Embedding Dim` | `32` | Dimension of the AST Node Type embeddings. |
| `Hidden Dim` | `64` | Dimension of the LSTM hidden state (Bi-LSTM results in `128`). |
| `Literal Dim` | `16` | Dimension of the literal feature vector. |
| `Fingerprint Dim`| `64` | Final L2-normalized output vector size. |

### Architecture Detail

1.  **Structural Channel (Bi-LSTM)**:
    - Inputs: `(Batch, 2048)` token IDs.
    - Logic: Processes the sequence of AST nodes in both directions to capture local context (e.g., being inside a `try/catch` or a specific `CallExpression`).
    - Pooling: Uses **Global Average Pooling** (ignoring padding) to create a fixed-size structural representation.

2.  **Literal Channel (Linear + Relu)**:
    - Inputs: `(Batch, 32)` numeric hashes (0.0 to 1.0).
    - Logic: Preserves the "semantic texture" of the code (e.g., specific error strings, math constants).
    - Pooling: Uses order-independent average pooling so that literal order doesn't affect the fingerprint.

3.  **Fusion Layer**:
    - Concatenates the structural summary and literal features.
    - Projects them through a final Linear layer to a 64-dimensional space.
    - Applies **L2 Normalization** to ensure that similarity can be calculated via a simple dot product.

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
