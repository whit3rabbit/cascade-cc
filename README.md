# Cascade-CC

Example output code download (see what the final output looks like): https://github.com/whit3rabbit/cascade-cc/releases/download/2.1.14/2.1.14.zip (release: https://github.com/whit3rabbit/cascade-cc/releases/tag/2.1.14)

A high-performance deobfuscation engine for Claude Code bundles using **Hybrid Differential Deobfuscation**.
This project is very loosely based on the Cascade JS Deobfuscator.

The system combines **Graph Theory** (Markov Centrality), **Neural Fingerprinting** (Siamese Transformer Networks), and **LLM Semantic Inference** to reconstruct minified code into its original, human-readable source structure.

---

## Quick Start (Just Cloned?)

Follow these steps to set up the environment and initialize the pre-trained "Brain."

### 1. Environment Setup
```bash
# Install Node dependencies
npm install

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure API Keys (OpenRouter or Gemini)
cp .env.example .env
```

### 2. Initialize the Pre-Trained Registry
Since the repo includes a pre-trained `ml/model.pth`, you just need to generate the logic vectors for the standard libraries:
```bash
npm run sync-vocab
npm run bootstrap
./sync_registry.sh
```

---

## Standard Analysis Workflow

Once initialized, use this sequence to analyze any new version of Claude:

```bash
# 1. Fetch the latest Claude bundle and chunk it
npm run analyze
#npm run analyze -- --version <version>

# 2. Anchor and classify roles (Replace <version> with output from step 1)
npm run anchor-classify -- <version>

# 4. LLM Deobfuscation (Focuses only on proprietary "Founder" logic)
npm run deobfuscate -- <version> --skip-vendor

# 5. Assemble into a structured codebase
npm run assemble -- <version>

# 6. (Optional) Final Logic Refinement Pass
npm run refine -- <version>
```

---

## Improving Results with "Custom Gold" Code

If you have manually deobfuscated a file, you can "teach" the Neural Network to recognize it. This turns your manual work into a template that will **automatically deobfuscate** that logic in all future versions.

### How to add your custom code:
1.  **Drop** your deobfuscated `.ts` or `.js` files into `ml/custom_gold/`.
    - You can optionally organize by version: `ml/custom_gold/0.2.8/`, `ml/custom_gold/0.2.9/`, etc.
    - Folder structure is preserved and used as a path hint during anchoring.
2.  **Ingest & Sync**:
    ```bash
    # Prepares the files for the registry
    node src/ingest_custom_gold.js
    
    # Vectorizes your code using the Brain and updates the registry
    ./sync_registry.sh
    ```
3.  **Result**: The next time you run `npm run anchor`, the system will identify your proprietary logic with `1.0` confidence and auto-rename variables/functions before the LLM pass.
4.  **Update KB hints** (optional): regenerate `knowledge_base.json` from your custom gold sources.
    ```bash
    # Backs up knowledge_base.json to knowledge_base.json.bak, then generates a new KB
    node scripts/generate_kb_from_gold.js
    ```

### Generate Knowledge Base From Custom Gold
If you want KB hints (file anchors, name hints, error anchors) derived from your `ml/custom_gold` sources:
```bash
# Backs up knowledge_base.json to knowledge_base.json.bak, then generates a new KB
node scripts/generate_kb_from_gold.js
```
Notes:
- The generator preserves `known_packages` and `structural_anchors` from the backup so import refinement keeps working.
- `project_structure` is auto-built from the gold folder tree; descriptions are generic.

> [!IMPORTANT]
> If you add a significant amount of new code, consider **retraining from scratch** (see below) to help the model learn the unique structural "DNA" of the new logic.

---

## Training the "Brain" (Optional)

The model in `ml/model.pth` is a specialized Transformer Encoder trained to be "name-blind"—it learns logic topology rather than syntax.

### Hardware-Aware Auto-Scaling
The training pipeline automatically adjusts the **Context Window** (`MAX_NODES`) based on your VRAM:

| Environment | Hardware | Default Window |
| :--- | :--- | :--- |
| **High-End GPU** | A100 / H100 (>15GB VRAM) | **2048 nodes** |
| **Standard GPU** | RTX 3080 / T4 (>7GB VRAM) | **1024 nodes** |
| **Apple Silicon** | Mac M1/M2/M3 (MPS) | **256 nodes** |
| **CPU Only** | Any | **512 nodes** |

### Retraining Instructions:
```bash
# Retrain from scratch to incorporate Custom Gold logic
npm run train -- --epochs 100 --batch_size 24 --device auto

# IMPORTANT: Always re-sync the registry after training a new model
./sync_registry.sh
```

---

## Key Concepts

### Founder vs. Vendor Code
*   **Vendor**: Standard libraries (React, Zod, Lodash). The NN identifies these with ~98% accuracy.
*   **Founder**: Claude's proprietary logic. If the NN hasn't seen it, the **Architectural Classifier** identifies it as "Founder Core" based on its centrality in the graph and sends it to the LLM.

### The "Cold Start" Advantage
Because the NN is trained on structural patterns, it can identify library versions it has **never seen before**, provided they follow similar architectural patterns to other libraries in the training set.

### Incremental Knowledge Transfer
When Claude releases a new version (e.g., `2.1.7` -> `2.1.9`), you can transfer deobfuscation knowledge from the old version to the new one:
```bash
# Anchor the new version using the old one as a reference
node run anchor 2.1.9 2.1.7
```

---

## Project Architecture

- `src/`: JavaScript core (Babel renaming, Chunking, Orchestration).
- `ml/`: Python ML core (Siamese Transformer training, Vectorization).
- `cascade_graph_analysis/`: Project metadata, Logic DB, and mappings.
- `docs/`: [Architecture](docs/ARCHITECTURE.md), [NN Internals](docs/NN.md), [Schema](docs/SCHEMA.md).
- `visualizer/`: D3.js-based interactive dependency graph (`npm run visualize`).
