require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { jsonrepair } = require('jsonrepair');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

const VERSION = '2.1.12';
const VERSION_DIR = `./cascade_graph_analysis/${VERSION}`;
const SYSTEM_MAP_PATH = path.join(VERSION_DIR, 'system_map.v2.json');
const OUTPUT_ROOT = path.join(VERSION_DIR, 'refined_codebase');

async function loadSystemMap() {
    if (!fs.existsSync(SYSTEM_MAP_PATH)) {
        throw new Error(`System Map not found at ${SYSTEM_MAP_PATH}`);
    }
    return JSON.parse(fs.readFileSync(SYSTEM_MAP_PATH, 'utf8'));
}

async function refineCodebase() {
    console.log(`[*] Starting Systematic Refinement for version ${VERSION}...`);
    const systemMap = await loadSystemMap();

    const registry = systemMap.symbol_registry;
    const modules = systemMap.modules;

    if (!fs.existsSync(OUTPUT_ROOT)) fs.mkdirSync(OUTPUT_ROOT, { recursive: true });

    for (const [proposedPath, meta] of Object.entries(modules)) {
        console.log(`[*] Refining module: ${proposedPath}`);

        let mergedCode = "";
        for (const chunkName of meta.original_chunks) {
            const chunkPath = path.join(VERSION_DIR, 'chunks', chunkName);
            if (fs.existsSync(chunkPath)) {
                mergedCode += fs.readFileSync(chunkPath, 'utf8') + "\n";
            } else {
                console.warn(`[!] Chunk not found: ${chunkName}`);
            }
        }

        // Perform scope-aware replacement using Babel bindings
        let refinedCode = mergedCode;
        try {
            const ast = parser.parse(mergedCode, {
                sourceType: 'module',
                plugins: ['jsx', 'typescript'],
                allowUndeclaredExports: true
            });

            const renamedBindings = new WeakSet();
            traverse(ast, {
                Identifier(p) {
                    const obf = p.node.name;
                    const entry = registry[obf];
                    if (!entry) return;
                    const refined = typeof entry === 'string' ? entry : entry.refined;
                    if (!refined || typeof refined !== 'string' || refined === obf) return;

                    const binding = p.scope.getBinding(obf);
                    if (!binding || renamedBindings.has(binding)) return;

                    try {
                        p.scope.rename(obf, refined);
                        renamedBindings.add(binding);
                    } catch (err) {
                        console.warn(`[!] Failed to rename ${obf} -> ${refined}: ${err.message}`);
                    }
                }
            });

            refinedCode = generate(ast, { retainLines: true }).code;
        } catch (err) {
            console.warn(`[!] Babel refine failed for ${proposedPath}: ${err.message}. Falling back to original code.`);
        }

        // Construct Output Dir
        const fullOutputPath = path.join(OUTPUT_ROOT, proposedPath);
        const outputDir = path.dirname(fullOutputPath);
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

        // Save Refined Output
        fs.writeFileSync(fullOutputPath, refinedCode);
        console.log(`[+] Saved refined module to ${fullOutputPath}`);
    }

    console.log(`[+] Systematic Refinement complete. Output in ${OUTPUT_ROOT}`);
}

refineCodebase().catch(err => {
    console.error(`[!] Refinement failed: ${err.message}`);
    process.exit(1);
});
