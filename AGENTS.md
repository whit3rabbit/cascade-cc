# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core analysis, LLM pipeline, and anchoring/renaming logic.
- `ml/`: Machine Learning components (PyTorch) for structural fingerprinting.
- `visualizer/`: WebGL graph viewer (served by `npm run visualize`).
- `run.js`: Command dispatcher used by `node run <task>`.
- `knowledge_base.json`: Seed terms for chunk identification.
- `cascade_graph_analysis/`: Generated outputs; treat as build artifacts.
- `claude-analysis/`: Sample input bundles and analysis references.

## File Structure (Key Files)
- `src/analyze.js`: Bundle analysis, simplified AST generation, and centrality scoring.
- `src/anchor_logic.js`: ML-backed structural similarity matching and mapping.
- `src/assemble_final.js`: Assemble deobfuscated chunks into final codebase.
- `src/bootstrap_libs.js`: Bootstrap vendor/runtime library detection.
- `src/deobfuscate_pipeline.js`: LLM pipeline and resume-aware orchestration.
- `src/deobfuscation_helpers.js`: Shared helpers for deobfuscation steps.
- `src/init_registry.js`: Initialize naming/metadata registries.
- `src/llm_client.js`: LLM provider selection, retries, and request plumbing.
- `src/rename_chunks.js`: Scope-aware Babel renames for deobfuscated chunks.
- `src/sync_vocab.js`: Sync vocabulary/term lists for analysis.
- `src/update_registry_from_bootstrap.js`: Update registry from bootstrap data.
- `ml/constants.py`: Shared ML constants and configuration.
- `ml/encoder.py`: Model definition for structural fingerprinting.
- `ml/model.pth`: Trained model weights.
- `ml/train.py`: Training loop for the structural fingerprint model.
- `ml/vectorize.py`: Vectorization pipeline for chunk logic embeddings.
- `visualizer/app.js`: WebGL graph renderer logic.
- `visualizer/index.html`: Visualizer entry point.
- `visualizer/style.css`: Visualizer styles.
- `docs/ARCHITECTURE.md`: System architecture overview.
- `docs/NN.md`: Neural network internals.
- `docs/SCHEMA.md`: JSON schema documentation.

## Build, Test, and Development Commands
- `source .venv/bin/activate`: Activate the local Python virtualenv (macOS).
- `npm install`: Install dependencies.
- `npm start`: Run the analyzer (defaults to `./cli.js` or fetches latest bundle).
- `npm run analyze -- path/to/bundle.js`: Analyze a specific bundle (generates `simplified_asts.json`).
- `npm run anchor -- <target> <reference>`: Structural similarity matching via Python NN.
- `npm run deobfuscate -- [version] [--limit N] [--skip-vendor] [--force] [--rename-only]`: Run the LLM pipeline and/or rename pass.
- `npm run assemble -- <version>`: Assemble deobfuscated chunks into a final file structure.
- `npm run visualize`: Start the local visualizer.
- `npm run lint`: Format all files with Prettier.

## Coding Style & Naming Conventions
- Formatting is handled by Prettier (`npm run lint`).
- File names in `src/` use lowercase with underscores (e.g., `deobfuscate_pipeline.js`).

## Testing Guidelines
- No automated test suite is currently present; validate with `npm start` and check `cascade_graph_analysis/`.

## Architecture Overview
- Claude Code bundles are esbuild-style single files (`cli.js`); `src/analyze.js` uses `webcrack` (with `unminify: true`), detects runtime helpers/`INTERNAL_STATE`, and exports **name-agnostic simplified ASTs** for ML.
- **Structural Fingerprinting**: `src/anchor_logic.js` bridges to PyTorch models in `ml/` to generate logic embeddings for code chunks.
- `src/analyze.js` calculates **centrality scores** using a Markov Chain (with `0.85` damping factor) to identify the "brain" of the application.
- The deobfuscation pipeline in `src/deobfuscate_pipeline.js` processes chunks by **Centrality order** and **Category priority** (Founder > Family > Vendor).
- Implements a **Resume Mechanism** via `processed_chunks` in `mapping.json` to avoid re-processing chunks. Use `--force` to override.
- LLM prompts include **Neighbor Roles/DisplayNames** and use a **Candidate Pool** filter to minimize token usage and redundant naming.
- `src/rename_chunks.js` applies Babel renames with refined **scope-aware logic** (inclusive of esbuild wrappers), producing `deobfuscated_chunks/`.
- `src/analyze.js` writes chunk files to `chunks/` and graph/centrality metadata to `metadata/graph_map.json`.
- The deobfuscation ("decryptioning") pipeline in `src/deobfuscate_pipeline.js` calls LLMs via `src/llm_client.js` and saves `metadata/mapping.json`.
- `src/assemble_final.js` (run via `npm run assemble`) performs Path-First Aggregation to create `final_codebase/`.
- `visualizer/` reads `metadata/` plus chunks.

## Detailed Documentation
For more in-depth information about the system, please refer to:
- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed breakdown of the analysis and deobfuscation pipeline.
- [Neural Network (NN) Internals](docs/NN.md) - Deep dive into the machine learning models and vectorization logic.
- [Data Schemas](docs/SCHEMA.md) - Documentation for the JSON schemas used throughout the project.

```mermaid
flowchart TD
  A[claude-analysis/bundle.js] --> B[src/analyze.js]
  B --> C[metadata/simplified_asts.json]
  C --> D[src/anchor_logic.js]
  D --> E[ml/vectorize.py]
  E --> F[metadata/logic_db.json]
  F --> G[src/anchor_logic.js]
  G --> H[metadata/mapping.json]
  H --> I[src/deobfuscate_pipeline.js]
  I --> J[metadata/mapping.json]
  J --> K[src/rename_chunks.js]
  K --> L[deobfuscated_chunks/]
  L --> M[src/assemble_final.js]
  M --> N[final_codebase/]
```

## Commit & Pull Request Guidelines
- Commit history favors short, imperative summaries (e.g., "update pipeline"). Keep messages concise.
- PRs should include: a clear description, how to reproduce/verify, and sample outputs.
- Include screenshots or a short clip for visualizer/UI changes.

## Configuration & Security Notes
- LLM runs require `.env` (see `.env.example`); set `LLM_PROVIDER`, `LLM_MODEL`, and an API key; never commit secrets.
- `src/llm_client.js` defaults to Gemini if unset, selects the matching key, and retries OpenRouter 429s with backoff.
- Use `GEMINI_API_KEY` for Gemini or `OPENROUTER_API_KEY` for OpenRouter; keys with `your_` are treated as invalid.
- Large generated outputs belong in `cascade_graph_analysis/` and should not be hand-edited.
- This repo assumes macOS for local development.
