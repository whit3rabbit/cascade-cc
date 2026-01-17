# Claude Code Cascade Analyzer

Pre-processor for CASCADE-style analysis of Claude Code bundles. 

Using a **Hybrid Differential Deobfuscation** approach (Graph Theory + Neural Fingerprinting + LLM Deobfuscation).

### Workflow

1. Bootstrap Library DNA - Download the libraries Claude depends on (Zod, React, etc.) and extract their structural fingerprints. Mangles and minifies the libraries to simulate real-world obfuscation. We train the neural network on this data. ```npm run bootstrap && npm run train```
2. Analyze & Anchor Claude - Analyze a real Claude bundle. This is a two-part process: **Structural Analysis** (JavaScript) followed by **Neural Anchoring** (Python). ```npm run analyze && npm run anchor -- <version>```
3. Deobfuscate (LLM Phase) - Process the proprietary "Founder" logic using the LLM. ```npm run deobfuscate -- <version>```
4. Assemble Final Codebase - Organize deobfuscated chunks into a coherent file structure based on inferred roles. ```npm run assemble -- <version>```
5. LLM Refinement Pass - Perform final logic reconstruction on the assembled codebase to restore original control flow and readability. ```npm run refine -- <version>```
6. LLM Refinement Pass - Perform final logic reconstruction on the assembled codebase to restore original control flow and readability. ```npm run refine -- <version>```
7. Interactive Visualization - View the dependency graph and Markov centrality scores. ```npm run visualize```

---

## TL;DR

Install prerequisites below. Then run the following commands:

```bash
1.  `npm run sync-vocab` # Sync the vocabulary
2.  `npm run bootstrap` # Download the libraries Claude depends on (Zod, React, etc.) and extract their structural fingerprints. Mangles and minifies the libraries to simulate real-world obfuscation. We train the neural network on this data.
3.  `node src/update_registry_from_bootstrap.js` # Update the logic registry with the bootstrap data
4.  `npm run train` # Optional, if you want to retrain the model
5.  `npm run analyze` # Analyze the Claude bundle
6.  `npm run anchor -- <version>` # Identify the libraries using the "Brain"
7.  `npm run deobfuscate -- <version> --skip-vendor` # Deobfuscate the proprietary "Founder" logic using the LLM
8.  `npm run assemble -- <version>` # Split the deobfuscated chunks into a coherent file structure based on inferred roles
9.  `npm run refine -- <version>` # Final LLM pass over the assembled codebase for a more readable output
10. `npm run visualize` # View the dependency graph and Markov centrality scores
```

## 1. Installation & Setup

### Requirements
- **Node.js** (v18+)
- **Python** (v3.10+)
- **OpenRouter/Gemini API Key** (for LLM deobfuscation)

```bash
# 1. Install Node dependencies
npm install

# 2. Configure Environment
cp .env.example .env # Add your API keys

# 3. Setup ML Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Getting Started (Cold Start Workflow)

If you have just cloned this repo, you can skip and use the pre-trained model in `ml/model.pth`. 

Here are instructions for building your own model if you don't want to use the pre-trained model.

### GPU Acceleration (NEW)

The ML pipeline now supports explicit device selection and uses a **Transformer Encoder** architecture, which thrives on GPU parallelization.

| Device | Platform | Recommendation |
| :--- | :--- | :--- |
| `mps` | Apple Silicon (M1/M2/M3) | **Best for Mac users.** |
| `cuda` | NVIDIA GPU | **Best for Linux/Windows servers.** |
| `cpu` | Any | Fallback (slower). |
| `auto` | Any | Automatically detect best available. |

#### Memory Considerations (OOM Troubleshooting)
The new Transformer architecture has $O(N^2)$ memory complexity relative to the sequence length (`MAX_NODES`).
- **Default (2048):** Stable on most systems with `batch_size=8`.
- **Large Context (4096):** May require `batch_size=1` or `2` on MPS (Mac) or CUDA (8GB VRAM).

**How to fix OOM on Mac (MPS):**
1. Reduce batch size: `npm run train -- --batch_size 2`
2. Set high watermark ratio (allows more system memory):
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   npm run train -- --batch_size 4
   ```
3. Revert to CPU: `npm run train -- --device cpu`

### Step 1: Bootstrap Library DNA

This downloads the libraries Claude depends on (Zod, React, etc.) and extracts their structural fingerprints. It loads libraries into `ml/bootstrap_data` and creates a logic registry in `cascade_graph_analysis/bootstrap`.

```bash
npm run bootstrap
```

### Step 2: Train the Neural Network (Optional)

This teaches the model to recognize the DNA of those libraries even when they are mangled/minified. This is optional and I have included a pre-trained model in the repository.

```bash
# Default (auto-detect)
npm run train

# Force Apple Metal (MPS)
npm run train -- --device mps --batch_size 8

# Force CUDA with custom batch size
npm run train -- --device cuda --batch_size 16
```
*Output: `ml/model.pth`. Your analyzer is now "primed" to recognize standard code.*


### Step 3: Analyze & Anchor Claude

Now, analyze a real Claude bundle. This is a two-part process: **Structural Analysis** (JavaScript) followed by **Neural Anchoring** (Python).

```bash
# 1. Analyze (JavaScript Phase)
# Fetches, de-proxies, and chunks the latest Claude version.
npm run analyze

# 2. Anchor (Python/NN Phase)
# Runs the Neural Network (ml/vectorize.py) to identify libraries using the "Brain".
# (Replace '2.1.7' with the version reported by the analyze step)
npm run anchor -- 2.1.7 --device auto

# 3. Deobfuscate (LLM Phase)
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
- `claude-analysis/`: Source bundles.
- `claude-analysis/`: Source bundles.
- `docs/`: [Detailed Architecture](docs/ARCHITECTURE.md), [NN Internals](docs/NN.md), [Schema Definitions](docs/SCHEMA.md), and [Environment Configuration](docs/ENVIRONMENT.md).

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
# 1. Analyze the new version
npm run analyze -- 2.1.9

# 2. Transfer knowledge from the old version
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