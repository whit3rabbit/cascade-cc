# Claude Code Cascade Analyzer

Pre-processor for CASCADE-style analysis of Claude Code bundles. 

Using a **Hybrid Differential Deobfuscation** approach (Graph Theory + Neural Net Fingerprinting using Transformers + LLM Deobfuscation).

### Workflow

1. Bootstrap Library DNA - Download the libraries Claude depends on (Zod, React, etc.) and extract their structural fingerprints. Mangles and minifies the libraries to simulate real-world obfuscation. We train the neural network on this data. ```npm run bootstrap && npm run train```
2. Analyze & Anchor Claude - Analyze a real Claude bundle, then anchor and re-run analyze to apply library/vendor classification from anchor metadata. ```npm run analyze && npm run anchor -- <version> && npm run analyze -- claude-analysis/<version>/cli.js --version <version>```
3. Deobfuscate (LLM Phase) - Process the proprietary "Founder" logic using the LLM. ```npm run deobfuscate -- <version>```
4. Assemble Final Codebase - Organize deobfuscated chunks into a coherent file structure based on inferred roles. ```npm run assemble -- <version>```
5. LLM Refinement Pass - Perform final logic reconstruction on the assembled codebase to restore original control flow and readability. ```npm run refine -- <version>```
6. Interactive Visualization - View the dependency graph and Markov centrality scores. ```npm run visualize```

> [!TIP]
> **Pro-Tip**: Use the `--device auto` flag with `npm run train` or `npm run anchor` to let the system choose between CUDA, MPS (Metal), or CPU automatically.

---

## TL;DR

Install prerequisites below. Then run the following commands:

### One Time Set Up

```bash
npm install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env 
# You must fill in LLM API Key in .env to either OpenRouter or Gemini if you want to run deobfuscation
```

Setting up bootstrap data

```bash
npm run sync-vocab
npm run bootstrap
node src/update_registry_from_bootstrap.js
```

### Typical Analysis Workflow

```bash
# 1. Fetch and chunk the latest Claude bundle
npm run analyze 

# 2. Identify libraries using the pre-trained Brain (Replace <version> with output from step 1)
# Requires ml/model.pth; run training if it's missing.
# npm run resync-registry # Troubleshooting if anchor fails with zero matches
npm run anchor -- <version> 

# 4. Role & Folder hinting
node src/classify_logic.js

# 6. Use LLM to name proprietary logic (Requires API Key in .env)
npm run deobfuscate -- <version> --skip-vendor

# 7. Assemble chunks into a final file structure
npm run assemble -- <version>

# 8. (Recommended) Restore original logic flow and readability via LLM
npm run refine -- <version>
```

## Training the "Brain" (Transformer Encoder)

If you have just cloned this repo, you can skip and use the pre-trained model in `ml/model.pth`.
Vectorization requires `ml/model.pth` to exist; training does not load it unless you pass `--finetune`.

Here are instructions for building your own model if you don't want to use the pre-trained model.

### Hardware-Aware Auto-Scaling (New)

The ML pipeline is now hardware-aware. It dynamically adjusts the **Context Window** (`MAX_NODES`) based on your detected VRAM to prevent Out-of-Memory (OOM) errors while maximizing performance on high-end hardware.

| Environment | Detected Hardware | Default Window | Recommendation |
| :--- | :--- | :--- | :--- |
| **Google Colab** | A100 / H100 (>15GB VRAM) | **2048 nodes** | Best for large-scale training. |
| **Standard GPU** | RTX 3080 / T4 (>7GB VRAM) | **1024 nodes** | Balanced performance. |
| **Mac (MPS)** | Apple Silicon (M1/M2/M3) | **256 nodes** | Optimized for unified memory. |
| **CPU Only** | Any | **512 nodes** | Safe fallback. |

#### Manual Overrides
You can manually override the auto-scaling using the `--max_nodes` flag:
```bash
# Force a massive window on a high-end server
npm run train -- --max_nodes 4096 --device cuda

# Force a tiny window for quick local testing
npm run train -- --max_nodes 128 --device cpu
```

### Realistic Logic Topology (Structural Noise)
The training process uses **Synthetic Mangling** with structural noise to ensure the model learns *logic* rather than just *syntax*.
- **IfStatement Swapping**: Negates and swaps `if/else` branches.
- **Commutative Swapping**: Reorders operands in `a + b` or `a && b`.
- **List Shuffling**: Reorders non-dependent statements in blocks.

### Step 1: Bootstrap Library DNA

This downloads the libraries Claude depends on (Zod, React, etc.) and extracts their structural fingerprints. It loads libraries into `ml/bootstrap_data` and creates a logic registry in `cascade_graph_analysis/bootstrap`.

```bash
npm run bootstrap
```

### Step 2: Train the Neural Network (Optional)

This teaches the model to recognize the DNA of those libraries even when they are mangled/minified. This is optional and I have included a pre-trained model in the repository.

#### Basic Training
```bash
npm run train -- --epochs 50 --batch_size 64 --device auto
```
Add `--finetune` if you want to resume from an existing `ml/model.pth`.

#### Hyperparameter Sweeps
If you want to find the absolute best settings for your hardware:
```bash
npm run train -- --sweep --epochs 3
```
*This will test multiple combinations of margin/learning-rate/embedding-dims and save the best model.*
*Output: `ml/model.pth`. Your analyzer is now "primed" to recognize standard code.*


### Step 3: Analyze & Anchor Claude

Now, analyze a real Claude bundle. This is a three-step process: **Structural Analysis** (JavaScript), **Neural Anchoring** (Python), then **Re-Analyze** to apply anchor metadata to classifications.

```bash
# 1. Analyze (JavaScript Phase)
# Fetches, de-proxies, and chunks the latest Claude version.
npm run analyze

# 2. Anchor (Python/NN Phase)
# Runs the Neural Network (ml/vectorize.py) to identify libraries using the "Brain".
# (Replace '2.1.7' with the version reported by the analyze step)
npm run anchor -- 2.1.7 --device auto

# 3. Re-Analyze (Apply Anchor Metadata)
# Uses the local bundle to avoid re-downloading.
npm run analyze -- claude-analysis/2.1.7/cli.js --version 2.1.7

# 4. Deobfuscate (LLM Phase)
# Processes the proprietary "Founder" logic using the LLM.
npm run deobfuscate -- 2.1.7 --skip-vendor
```

---

## 5. Post-Processing & Reference

### Assemble Final Codebase

Organize deobfuscated chunks into a coherent file structure based on inferred roles.

```bash
npm run assemble -- 2.1.7
```

### LLM Refinement Pass

Perform final logic reconstruction on the fully assembled codebase. This stage restores clean control flow (if/else), groups related functions, and removes any remaining obfuscation artifacts.

```bash
npm run refine -- 2.1.7
```

*Output: Refined source code is saved in `cascade_graph_analysis/2.1.7/refined_assemble/`.*

### Interactive Visualization

View the dependency graph and Markov centrality scores.

```bash
npm run visualize
# Open http://localhost:3000/visualizer/
```

### Project Architecture

- `src/`: JavaScript core (Babel renaming, Chunking, Orchestration).
- `ml/`: Python ML core (Triplet Loss training, Vectorization).
- `cascade_graph_analysis/`: Project metadata, Logic DB, and mappings.
- `claude_analysis/`: Source bundles.
- `docs/`: [Architecture](docs/ARCHITECTURE.md), [NN Internals](docs/NN.md), [Schema](docs/SCHEMA.md), and [Environment](docs/ENVIRONMENT.md).

---

## Architecture: Training vs. Reference

The `npm run bootstrap` command populates two different directories. They represent two different sides of the same coin: **Training** vs. **Reference**.

### 1. `ml/bootstrap_data` (The Training Lab)

**Purpose:** Input for the Neural Network's **Training Phase**. Holds `*_gold_asts.json` clean library patterns.

### 2. `cascade_graph_analysis/bootstrap` (The Logic Registry)

**Purpose:** Output of the analysis tool, used for the **Anchoring Phase**. Acts as a lookup table of known structural vectors.

| Feature | `ml/bootstrap_data` | `cascade_graph_analysis/bootstrap` |
| :--- | :--- | :--- |
| **Role** | **Input** for Training | **Output** of Analysis |
| **User** | Python NN (`train.py`) | JS Anchoring (`anchor_logic.js`) |
| **Key File** | `zod_gold_asts.json` | `logic_db.json` |

### Why do we need both?

The **NN** identifies that two pieces of code are structurally similar. The **Registry** provides the human-readable names assigned to those structures in the clean, non-obfuscated version.

---

## Structural DNA & Version Drift

The Neural Network produce **Vector Embeddings** (comparable patterns) rather than simple hashes. This allows it to handle version drift and minification through **Structural DNA**.

### 1. Vectors vs. Hashes

If one character changes, a **Hash** breaks. If logic changes slightly, a **Vector** simply moves a fraction in vector space.
*   **Minor Updates (e.g., Zod 4.3.4 -> 4.3.5):** The "backbone" remains the same. The resulting vector maintains **~98-99% similarity**.
*   **Synthetic Mangling:** In `ml/train.py`, we intentionally rename variables to nonsense during training. The NN learns to ignore **"Surface Noise"** and focus on **"Logic Topology"**.

### 2. Guardrails & Safety

In `src/anchor_logic.js`, we use similarity thresholds and symbol alignment to ensure accuracy:

| Similarity | Result | Action |
| :--- | :--- | :--- |
| **> 0.95** | High Confidence | Auto-rename everything (Relaxed from 0.98 for Cold Start) |
| **0.90 - 0.94** | Version Drift | Flag for LLM/Partial anchor |
| **< 0.85** | New Logic | Send to LLM |

### The "Cold Start" Advantage

By training on many libraries, the NN learns what **"Library-ness"** looks like. It can categorize vendor code even if it has never seen that specific version before, allowing the LLM to focus purely on the proprietary "Founder" logic.

---

## 7. Incremental Knowledge Transfer (Upgrading Versions)

When Claude releases a new version (e.g., `2.1.7` -> `2.1.9`), much of the underlying logic remains identical. You can "upgrade" your analysis and transfer deobfuscation knowledge from an earlier version to a new one.

### How to Upgrade

If you have already deobfuscated version `2.1.7` and want to analyze `2.1.9`, run:

```bash
# 1. Analyze the new version (downloads latest if not present)
npm run analyze

# 2. If you already have the bundle locally, you can re-run analyze without re-downloading
npm run analyze -- claude-analysis/2.1.9/cli.js --version 2.1.9

# 3. Transfer knowledge from the old version
node run anchor 2.1.9 2.1.7
```

### How Knowledge Transfer Works

The `anchor` command performs a direct version-to-version comparison:

1.  **Vectorization:** Both the new version (`2.1.9`) and the reference version (`2.1.7`) are vectorized.
2.  **Structural Matching:** The tool looks for high-similarity matches (>90%) between chunks. Because vectors focus on "Logic Topology," minor code shifts (renamed variables or reordered statements) don't break the match.
3.  **Symbol Alignment:** When a match is found, the tool uses **Structural Keys** (AST-based fingerprints) to align symbols between versions.
4.  **Mapping Injection:** Resolved names from the `2.1.7` `mapping.json` are injected into the `2.1.9` `mapping.json`.

**Benefits:**
*   **Token Savings:** Already deobfuscated logic is skipped in the next LLM pass.
*   **Consistency:** Maintains naming conventions across versions.
*   **Speed:** Only new or significantly changed logic needs fresh deobfuscation.
