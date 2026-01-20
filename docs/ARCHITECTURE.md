# Repository Architecture & AI Workflow

This document provides a detailed technical overview of the Claude Code Cascade Analyzer's architecture, its hybrid deobfuscation methodology, and the integrated AI workflow.

---

## 1. High-Level Methodology: Hybrid Differential Deobfuscation

The system employs a **Hybrid Differential Deobfuscation** approach. Instead of relying purely on pattern matching or LLMs, it combines three distinct domains to reconstruct obfuscated codebases:

1.  **Graph Theory & Static Analysis**: Understanding the relationship between code chunks, identifying entry points, and calculating importance (Markov Centrality/PageRank).
2.  **Neural Fingerprinting**: Using a **Transformer Encoder** (Siamese Network) to identify code based on its "Structural DNA" (AST topology) rather than literal content.
3.  **LLM Semantic Inference**: Using Large Language Models (LLMs) to reconstruct human-readable names and file structures for proprietary ("Founder") logic that lacks a public baseline.

---

## 2. System Components

The repository is partitioned into three main layers:

### A. Orchestration & AST Processing (Node.js)
- **`src/analyze.js`**: The core analyzer. It uses Babel to chunk large bundles, extract metadata, detected "Module Envelopes" for logical grouping, and calculates Identifier Affinity scores.
- **`src/anchor_logic.js`**: Bridges Node.js orchestration with Python inference. It handles logic registry synchronization, vectorization commands, and aligns symbols between target and gold-standard chunks.
- **`src/classify_logic.js`**: Assigns architectural roles and proposed paths based on graph metrics and anchor metadata.
- **`src/deobfuscate_pipeline.js`**: Manages the multi-stage LLM pass, including the **Consolidation Pass** for grouped chunks and inherited scope mapping.
- **`src/rename_chunks.js`**: Consumes `mapping.json` to apply a safe, scope-aware rename pass across the entire AST.
- **`src/assemble_final.js`**: The reconstruction engine that performs **Deduplicating Merges** to strip module wrappers and reassemble split chunks into clean files.

### B. Machine Learning Core (Python/PyTorch)
- **`ml/encoder.py`**: Defines the `TransformerCodeEncoder`. It uses a Transformer Encoder with Positional Encoding to generate fixed-size (32-dim) embeddings from AST sequences.
- **`ml/train.py`**: Implements the Triplet Loss training loop with "Nuclear Options" (Literal Dropout, Sequence Jittering, Node Masking) to force structural learning.
- **`ml/vectorize.py`**: Generates 32-dimensional L2-normalized embeddings (fingerprints) for code chunks.

### C. Knowledge & Metadata (JSON)
- **`knowledge_base.json`**: A curated database of known library "anchors" (keywords, error strings, unique patterns).
- **`logic_registry.json`**: A persistent database mapping structural vectors to confirmed library names and symbols.
- **`mapping.json`**: The version-specific result of the deobfuscation process, tracking every renamed identifier and its confidence score.

---

## 3. The AI Workflow (Step-by-Step)

```mermaid
flowchart TD
  subgraph "Phase 1: Bootstrapping & Training"
    B1[Bootstrap Libs] --> B2[Extract Gold ASTs]
    B2 --> B3[Train Transformer]
    B3 --> B4[ml/model.pth]
  end

  subgraph "Phase 2: Analysis & Neural Anchoring"
    A1[Analyze Target Bundle] --> A2[Chunking & Graph Mapping]
    A2 --> A3[Vectorize Chunks]
    A3 --> A4[Neural Anchoring]
    A4 --> A5[logic_registry.json / KB Match]
  end

  subgraph "Phase 2.5: Architectural Classification"
    A5 --> C1[Architectural Classification]
    C1 --> C2[Role & Path Assignment]
  end

  subgraph "Phase 3: LLM & Assembly"
    L1[Anchor Core Libs] --> L2[LLM Pass (Founder Logic)]
    L2 --> L3[mapping.json Update]
    L3 --> L4[Final Assembly]
  end

  subgraph "Phase 4: Refinement"
    R1[Refine Codebase] --> R2[Final Clean Source]
  end

  B4 -.-> A3
  C2 --> L1
  L4 --> R1
```

### Step 1: Bootstrapping Library DNA
The system downloads known libraries (e.g., React, Zod, Anthropic SDK) and generates clean, "Gold Standard" structural fingerprints. This creates the baseline "Brain" of the system.

### Step 2: Training (Nuclear Regularization)
The Transformer Network is trained on the bootstrapped data using **Triplet Loss**. To prevent the model from overfitting to easy signals (like string literals), we employ "Nuclear Options":
- **Literal Dropout**: Randomly masking all string/numeric literals.
- **Sequence Jittering**: Randomly cropping or padding sequences.
- **Node Type Masking**: Replacing valid AST node types with `UNKNOWN` tokens.

This forces the model to learn the *topology* of the logic (how nodes connect) rather than superficial markers.

### Step 3: Analysis & Chunking
The target bundle is broken down into small, manageable "chunks." The system calculates **Markov Centrality** (PageRank with 0.85 damping) to determine which chunks are the "heart" of the application. It uses "Hybrid Teleportation" to bias importance towards suspected Founder logic.

### Step 4: Neural Anchoring
Every chunk in the target bundle is vectorized. These vectors are compared against the `logic_registry.json`.
- **Similarity > 0.95**: Gold Standard / Library Match.
- **Similarity > 0.80**: Strong Structural Match.
- **Heuristic Boosts**: If `knowledge_base.json` suggests a library (e.g., "ink"), the similarity score is boosted to help the model lock in.

### Step 4.5: Architectural Classification
The classifier (`src/classify_logic.js`) uses graph metrics and anchor metadata to assign roles (e.g., `VENDOR_LIBRARY`, `APP_LOGIC`) and propose paths before the LLM pass.

### Step 5: LLM Deobfuscation
Chunks identified as "Founder" logic are sent to an LLM. The pipeline prioritizes "Core" chunks first. The LLM receives:
1.  The chunk's code.
2.  Neighboring chunk names (context).
3.  Any identified "Gold" partial matches.
4.  Existing mappings (incremental context).

### Step 6: Reconstruction
The final step uses the `mapping.json` to perform a scope-safe rename across all chunks via `src/rename_chunks.js` and writes them to a new directory structure that mirrors the inferred original codebase.

### Step 7: Logic Refinement
The assembled codebase undergoes a final refinement pass (`npm run refine`). This stage uses an LLM to restore high-level control flow (converting complex ternary chains back to if/else blocks) and remove lingering obfuscation boilerplate.

**Refinement scripts**:
- **`src/systematic_refiner.js`**: Bulk symbol replacement using the registry and mapping metadata.
- **`src/refine_codebase.js`**: LLM-driven logic reconstruction, aliased to `npm run refine`.

---

## 4. Advanced Chunk Reconstruction (The "Holy Grail")

To solve the problem of identifying which split chunks belong to the same original source file, the system uses a multi-tiered signal detection system.

### A. Hard Signal: Module Envelopes
The analyzer (`src/analyze.js`) detects bundler wrappers (e.g., `__commonJS`, `__lazyInit`) and assigns a persistent `moduleId` to all chunks contained within the wrapper. This effectively groups split parts of a large module together.

### B. Soft Signal: Identifier Affinity
We calculate a **Cohesion Score** between adjacent chunks based on the continuity of short, scoped variables (e.g., `_a`, `x`). If Chunk A defines `_a` and Chunk B immediately uses it, an **Affinity Link** is established, treating them as a logical continuum.

### C. Consolidation & Deduplication
- **Consolidation Pass**: Before deobfuscation, a "Consolidation Pass" groups linked chunks and prompts the LLM to identify the single "Unified Path" for the group.
- **Deduplicating Merge**: During assembly (`src/assemble_final.js`), module wrappers and redundant helpers are stripped, merging the bodies of split chunks into a clean, singular file.

*For a deep dive on the math and logic, see [Advanced Chunk Reconstruction](CHUNKS.md).*

---

## 5. Key Architectural Innovations

### The "Cold Start" Advantage
Because the NN is trained on structural patterns rather than specific hashes, it can identify a library version it has never seen before, provided it follows similar architectural patterns to other libraries in the training set.

### Literal Hashing & Dropout
The ML model uses a separate channel for literals (strings, numbers). During training, this channel is frequently "dropped out" (masked). This teaches the model to use literals as helpful hints when available, but not to depend on them, making it robust against string encryption or modification.

### Graph-Aware Deobfuscation
Most deobfuscators look at files in isolation. This system uses the dependency graph to propagate information. If `chunkA` is identified as "Gemini Client," and `chunkB` depends heavily on it, the LLM is primed to know that `chunkB` likely contains core AI orchestration logic.

---

## 6. Visualization Integration

The `visualizer/` directory contains a D3.js-based interface that allows developers to interactively explore the codebase's "Topology."
- **Nodes**: Chunks of code.
- **Edges**: Import/Export relationships.
- **Size**: Proportional to Markov Centrality (Importance).
- **Color**: Indicates whether a chunk is "Vendor" (Library) or "Founder" (App Logic).
