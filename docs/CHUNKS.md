# Advanced Chunk Reconstruction & Association

The "Holy Grail" of bundle deobfuscation is correctly identifying which split chunks belong to the same original source file. To solve this, Cascade employs a multi-tiered signal detection system that moves beyond simple token counting.

## 1. Hard Signal: The Module Envelope
Modern bundlers (like `esbuild` or `webpack`) often wrap modules in closure functions to manage scope. Even if a large file is split into multiple AST chunks during analysis, these chunks are physically contained within a single wrapper function in the original bundle.

**Detection Mechanism**:
The analyzer tracks specific runtime helpers (e.g., `__commonJS`, `__lazyInit`) that serve as "Module Envelopes."
-   When a chunk begins with a module envelope declaration, it is assigned a unique `moduleId`.
-   Subsequent chunks that fall physically within this wrapper (before the closing brace) inherit this `moduleId`.
-   **Result**: This provides a definitive, "hard" link between chunks, guaranteeing they share the same origin.

## 2. Soft Signal: Identifier Affinity (Scope Continuity)
When code is minified, top-level variables often get short, consistent names (e.g., `_a`, `x`, `Q`) effective only within that file's scope. If Chunk A defines a variable `_a` and Chunk B immediately uses `_a` without redefining it, they are inextricably linked by scope.

**The Math of Affinity**:
We calculate a **Cohesion Score** between adjacent chunks $C_i$ and $C_{i+1}$:

$$
Score(C_i, C_{i+1}) = | \{ v \in \text{Defined}(C_i) \} \cap \{ v \in \text{Used}(C_{i+1}) \} |
$$

Where:
-   $\text{Defined}(C_i)$ is the set of short (< 3 char) variables defined in the preceding chunk.
-   $\text{Used}(C_{i+1})$ is the set of short variables referenced in the succeding chunk.

If $Score \ge 3$, the system establishes an **Affinity Link**, treating $C_{i+1}$ as a direct continuation of $C_i$.

## 3. The Consolidation Pass (Phase 0.9)
Before individual deobfuscation, the pipeline runs a **Consolidation Pass**. It groups sequential chunks that share either a `moduleId` (Hard Link) or a high Affinity Score (Soft Link).

**The "Unified Path" Prompt**:
Instead of asking the LLM to name Chunk 1 and Chunk 2 separately, we feed the summary of the entire group:
> "These 5 chunks share a module scope. Identify the single original file they belong to (e.g., `src/services/auth.ts`) and assign them Part IDs."

## 4. Deduplicating Assembly
When reassembling the final codebase, simply concatenating chunks would result in repetitive artifacts (e.g., multiple `var wrapper = __commonJS(...)` headers).

**Structural Merge Logic**:
-   **Header Stripping**: The assembler detects and removes the module envelope definition from the first chunk of a group.
-   **Footer Stripping**: The closing braces of the wrapper are removed from the final chunk.
-   **Body Fusion**: The effective bodies of the chunks are merged, reconstructing the original file content without the bundler's scaffolding.
-   **Cycle Handling**: `src/assemble_final.js` uses Strongly Connected Components (SCC) logic to safely merge cyclic dependencies between chunks.
