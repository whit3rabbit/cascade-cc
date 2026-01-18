require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { jsonrepair } = require('jsonrepair');

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

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function refineCodebase() {
    console.log(`[*] Starting Systematic Refinement for version ${VERSION}...`);
    const systemMap = await loadSystemMap();

    const registry = systemMap.symbol_registry;
    const modules = systemMap.modules;

    // Create a mapping of all symbols to be replaced
    // We sort by length descending to avoid partial matches (e.g., replacement of 'abc' before 'abcd')
    const sortedSymbols = Object.keys(registry).sort((a, b) => b.length - a.length);

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

        // Perform Systematic Replacement
        let refinedCode = mergedCode;
        for (const obf of sortedSymbols) {
            const refined = registry[obf].refined;
            // Use word boundary to avoid accidental partial matches in larger strings
            const regex = new RegExp(`\\b${escapeRegExp(obf)}\\b`, 'g');
            refinedCode = refinedCode.replace(regex, refined);
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
