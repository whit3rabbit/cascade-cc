# Claude Code Cascade Analyzer

Pre-processor for CASCADE-style analysis of Claude Code bundles.

## Setup

Install dependencies:

```bash
npm install
```

## Usage

### Analyze

Run the analysis on a bundle. It defaults to searching for `./cli.js` or downloading the latest version from npm if not found.

```bash
npm start
```

To specify a path:

```bash
node analyze.js path/to/bundle.js
```

### Visualize

Start the interactive graph visualizer (WebGL powered by Sigma.js):

```bash
npm run visualize
```

Then open: [http://localhost:3000/visualizer/](http://localhost:3000/visualizer/)

## Workflow

The analysis process follows a multi-stage pipeline designed to decrypt and structure minified Claude code:

1.  **Phase 0: Preprocessing (Webcrack)**: The target bundle is processed via `webcrack` to unminify code, resolve shorthand arithmetic, and identify logical module boundaries.
2.  **Phase 1: AST identification**: The unminified code is parsed into an Abstract Syntax Tree (AST). Heuristics (inspired by `JSimplifier`) are used to split the tree into logical functional chunks (~1000+ modules) based on module wrappers, keyword signals, and utility patterns.
3.  **Phase 2: Neighbor Detection**: The analyzer builds a dependency graph by detecting cross-references between chunks. It tracks which chunks define internal names and which chunks reference them.
4.  **Phase 3: Centrality Calculation**: A Markov Chain analysis is applied to the graph to calculate the relative importance (centrality) of each chunk, helping identify the core logic of the Claude "agent".
5.  **Phase 4: Final Classification**: Chunks are categorized as `priority` (Claude logic) or `vendor` (library code) based on signal keywords and connectivity.

## Output

Results are saved to `cascade_graph_analysis/`:
- `chunks/`: Extracted functional chunks.
- `metadata/graph_map.json`: Graph structure and Markov centrality scores.

## Claude Code Obfuscation Analysis

An analysis of `claude-analysis/2.1.3/cli.js` (11MB bundle) reveals the following technical characteristics:

### Obfuscation Level
The code is **minified but not aggressively obfuscated**. It does not use advanced techniques like encrypted string tables, control flow flattening, or self-defending code. 

### Key Characteristics
- **Bundler**: Built using `esbuild`, identified by standard helper functions (`w`, `U`, `U5`) and lazy-load module initialization patterns.
- **Minification**: Local variable and function names are mangled (e.g., `hN9`, `_CA`), but high-level logic structures—such as classes and core control flows—remain discernable.
- **Dependency Aggregation**: The 11MB file is a standalone bundle containing numerous integrated dependencies, including `lodash`, `rxjs`, and various Node.js polyfills.
- **Internal Logic**: Logic for components like the native host (Chrome integration) and MCP (Model Context Protocol) clients is clearly visible as distinct classes (e.g., `Mz9`, `Rz9`).
- **Transparency**: Includes clear metadata such as build timestamps, versioning info, and even a recruitment message in the comments, suggesting the minification is for efficiency rather than secrecy.
