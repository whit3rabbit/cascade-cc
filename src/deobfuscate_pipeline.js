require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { callLLM, PROVIDER, MODEL } = require('./llm_client');

const OUTPUT_ROOT = './cascade_graph_analysis';

// --- KNOWLEDGE BASE ---
const KB_PATH = './knowledge_base.json';
let KB = null;
if (fs.existsSync(KB_PATH)) {
    KB = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
    console.log(`[*] Loaded Knowledge Base with ${KB.name_hints?.length || 0} name hints.`);
}

// --- UTILS ---
function getLatestVersion(baseDir) {
    if (!fs.existsSync(baseDir)) return null;
    const dirs = fs.readdirSync(baseDir).filter(f => fs.statSync(path.join(baseDir, f)).isDirectory());

    // Filter folders that look like semver (e.g., 2.1.3)
    const semverDirs = dirs.filter(d => /^\d+\.\d+\.\d+/.test(d));
    if (semverDirs.length === 0) return null;

    // Sort semver-like strings correctly
    semverDirs.sort((a, b) => {
        const partsA = a.split('.').map(Number);
        const partsB = b.split('.').map(Number);
        for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
            const valA = partsA[i] || 0;
            const valB = partsB[i] || 0;
            if (valA !== valB) return valB - valA; // Descending
        }
        return 0;
    });

    return semverDirs[0];
}

function extractIdentifiers(code) {
    // Simple regex-based identifier extraction
    // Matches common JS variable names, avoids some keywords
    const matches = code.match(/[a-zA-Z_$][a-zA-Z0-9_$]*/g) || [];
    const unique = new Set(matches);
    const keywords = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'enum', 'await', 'async', 'null', 'true', 'false', 'undefined']);
    return Array.from(unique).filter(id => !keywords.has(id));
}

// --- MAIN STAGES ---
async function run() {
    let version = process.argv[2];

    if (!version) {
        version = getLatestVersion(OUTPUT_ROOT);
        if (!version) {
            console.error(`[!] No analysis found in ${OUTPUT_ROOT}.`);
            console.error(`    Please run 'npm start' first to analyze the codebase.`);
            process.exit(1);
        }
        console.log(`[*] No version specified. Auto-detecting latest version: ${version}`);
    } else {
        console.log(`[*] Target version specified: ${version}`);
    }

    const versionPath = path.join(OUTPUT_ROOT, version);
    if (!fs.existsSync(versionPath)) {
        console.error(`[!] Error: Version directory not found at ${versionPath}`);
        console.error(`    Available versions: ${fs.readdirSync(OUTPUT_ROOT).filter(f => fs.statSync(path.join(OUTPUT_ROOT, f)).isDirectory()).join(', ') || 'None'}`);
        process.exit(1);
    }

    const chunksDir = path.join(versionPath, 'chunks');
    const mappingPath = path.join(versionPath, 'metadata', 'mapping.json');
    const deobfuscatedDir = path.join(versionPath, 'deobfuscated_chunks');

    if (!fs.existsSync(chunksDir)) {
        console.error(`[!] Error: Chunks directory not found at ${chunksDir}`);
        process.exit(1);
    }

    if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

    let globalMapping = {};
    if (fs.existsSync(mappingPath)) {
        try {
            globalMapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
            console.log(`[*] Loaded global mapping with ${Object.keys(globalMapping).length} entries.`);
        } catch (err) {
            console.warn(`[!] Error reading mapping.json, starting fresh: ${err.message}`);
        }
    }

    const graphMapPath = path.join(versionPath, 'metadata', 'graph_map.json');
    let graphData = [];
    if (fs.existsSync(graphMapPath)) {
        try {
            graphData = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
        } catch (err) {
            console.error(`[!] Error parsing graph_map.json: ${err.message}`);
            process.exit(1);
        }
    } else {
        console.error(`[!] Error: graph_map.json not found at ${graphMapPath}`);
        process.exit(1);
    }

    // Sort by centrality (High to Low)
    const sortedChunks = graphData.slice().sort((a, b) => b.centrality - a.centrality);

    console.log(`[*] Starting Deobfuscation Pipeline [Provider: ${PROVIDER}, Model: ${MODEL}]`);
    console.log(`[*] Processing ${sortedChunks.length} chunks by Centrality order.`);

    // Stage 1: Incremental Mapping
    for (let i = 0; i < sortedChunks.length; i++) {
        const chunkMeta = sortedChunks[i];
        const file = path.basename(chunkMeta.file);
        const chunkPath = path.join(chunksDir, file);

        if (!fs.existsSync(chunkPath)) {
            console.warn(`[!] Skip: Chunk file not found: ${chunkPath}`);
            continue;
        }

        const code = fs.readFileSync(chunkPath, 'utf8');
        const identifiers = extractIdentifiers(code);
        const filteredMapping = {};
        identifiers.forEach(id => {
            if (globalMapping[id]) filteredMapping[id] = globalMapping[id];
        });

        console.log(`[*] Stage 1 [${i + 1}/${sortedChunks.length}]: Naming Pass for ${file} (Centrality: ${chunkMeta.centrality.toFixed(4)})...`);

        const prompt = `
Role: Senior Reverse Engineer
Task: Semantic Variable Mapping

Analyze the provided JavaScript code chunk from an AI Agent CLI tool.
CHUNK METADATA:
- Role: ${chunkMeta.role}
- Label: ${chunkMeta.label}
- State DNA: This code interacts with global state properties: ${chunkMeta.state_touchpoints.join(', ')}
- Neighbors: Interacts with ${chunkMeta.outbound.join(', ')}

EXISTING MAPPINGS (Use these names for consistency):
${JSON.stringify(filteredMapping, null, 2)}

REFERENCE HINTS FROM KNOWLEDGE BASE:
${KB && KB.name_hints ? KB.name_hints.map(h => `- Logic: "${h.logic_anchor}" -> Suggested Name: "${h.suggested_name}"`).join('\n') : 'None'}

CODE:
\`\`\`javascript
${code}
\`\`\`

INSTRUCTIONS:
1. Identify the purpose of obfuscated variables based on their usage, logic, and metadata.
2. If a variable stores a value related to a State DNA property, Name it descriptively (e.g., 'sessionId' -> 'activeSessionId').
3. If the role is '${chunkMeta.role}', prioritize domain-specific names.
4. Return ONLY a JSON object of NEW or IMPROVED mappings. If you are updating an existing mapping, explain why in the rationale.
5. If the logic matches a REFERENCE HINT, you MUST use that suggested name.

RESPONSE FORMAT (JSON ONLY):
{
  "mappings": {
     "obfuscatedName": "descriptiveName"
  },
  "rationale": "Brief explanation of why these names were chosen."
}
`;

        try {
            const llmResponse = await callLLM(prompt);
            const jsonMatch = llmResponse.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const responseData = JSON.parse(jsonMatch[0]);
                const newMappings = responseData.mappings || {};

                let added = 0;
                for (const [key, val] of Object.entries(newMappings)) {
                    if (globalMapping[key] !== val) {
                        globalMapping[key] = val;
                        added++;
                    }
                }

                if (added > 0) {
                    fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
                    console.log(`    - Added/Updated ${added} mappings. Rationale: ${responseData.rationale}`);
                }
            } else {
                console.warn(`    [!] Could not parse JSON from LLM response for ${file}`);
            }
        } catch (err) {
            console.error(`    [!] Critical error processing ${file}: ${err.message}`);
            // We continue to the next chunk instead of crashing the whole process
        }
    }

    console.log(`[*] Stage 1 Complete. Final mapping preserved at ${mappingPath}`);

    // Stage 2: Safe Babel Renaming
    console.log(`[*] Starting Stage 2: Safe Renaming with Babel...`);
    try {
        const cmd = `node src/rename_chunks.js "${versionPath}"`;
        console.log(`[*] Executing: ${cmd}`);
        execSync(cmd, { stdio: 'inherit' });
        console.log(`\n[COMPLETE] Deobfuscation finished for ${version}`);
    } catch (err) {
        console.error(`[!] Stage 2 failed: ${err.message}`);
    }
}

run().catch(err => {
    console.error(`\n[FATAL ERROR] Pipeline aborted: ${err.message}`);
    process.exit(1);
});
