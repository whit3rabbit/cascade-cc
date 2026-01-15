# Claude Code Cascade Analyzer

Pre-processor for CASCADE-style analysis of Claude Code bundles. 

Using a **Hybrid Differential Deobfuscation** approach (Graph Theory + Neural Fingerprinting + LLM Deobfuscation).

### Workflow

1. Bootstrap Library DNA - Download the libraries Claude depends on (Zod, React, etc.) and extract their structural fingerprints. Mangles and minifies the libraries to simulate real-world obfuscation. We train the neural network on this data. ```npm run bootstrap && npm run train```
2. Analyze & Anchor Claude - Analyze a real Claude bundle. This is a two-part process: **Structural Analysis** (JavaScript) followed by **Neural Anchoring** (Python). ```npm run analyze && npm run anchor -- <version>```
3. Deobfuscate (LLM Phase) - Process the proprietary "Founder" logic using the LLM. ```npm run deobfuscate -- <version>```
4. Assemble Final Codebase - Organize deobfuscated chunks into a coherent file structure based on inferred roles. ```npm run assemble -- <version>```
5. Interactive Visualization - View the dependency graph and Markov centrality scores. ```npm run visualize```

---

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

### Step 1: Bootstrap Library DNA

This downloads the libraries Claude depends on (Zod, React, etc.) and extracts their structural fingerprints.

```bash
npm run bootstrap
```

### Step 2: Train the Neural Network

This teaches the model to recognize the DNA of those libraries even when they are mangled/minified.

```bash
npm run train
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
npm run anchor -- 2.1.7

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