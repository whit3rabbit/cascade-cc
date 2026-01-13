require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');

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
    // Enhanced extraction for variables and properties
    const variables = new Set();
    const properties = new Set();
    const keywords = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'enum', 'await', 'async', 'null', 'true', 'false', 'undefined']);
    const globals = new Set(['console', 'Object', 'Array', 'String', 'Number', 'Boolean', 'Promise', 'Error', 'JSON', 'Math', 'RegExp', 'Map', 'Set', 'WeakMap', 'WeakSet', 'globalThis', 'window', 'global', 'process', 'require', 'module', 'exports', 'URL', 'Buffer']);

    // Match variables (stand-alone identifiers)
    const idRegex = /\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g;
    let match;
    while ((match = idRegex.exec(code)) !== null) {
        const id = match[0];
        if (!keywords.has(id) && !globals.has(id)) {
            // IGNORE human-readable names: length > 4 or containing underscores or camelCase
            const isHumanReadable = id.length > 4 || id.includes('_') || (/[a-z]/.test(id) && /[A-Z]/.test(id));
            if (!isHumanReadable) {
                variables.add(id);
            }
        }
    }

    // Match potential property lookups: .prop or prop:
    const propRegex = /\.([a-zA-Z_$][a-zA-Z0-9_$]*)\b/g;
    while ((match = propRegex.exec(code)) !== null) {
        const prop = match[1];
        if (!keywords.has(prop) && !globals.has(prop) && prop.length > 1) {
            properties.add(prop);
        }
    }
    const propLitRegex = /\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:/g;
    while ((match = propLitRegex.exec(code)) !== null) {
        const prop = match[1];
        if (!keywords.has(prop) && !globals.has(prop) && prop.length > 1) {
            properties.add(prop);
        }
    }

    return {
        variables: Array.from(variables),
        properties: Array.from(properties)
    };
}

function filterKBHints(code, kb, maxHints = 100) {
    if (!kb || !kb.name_hints) return 'None';

    // Simple relevance check: does the logic anchor or suggested name share keywords with the code?
    const codeWords = new Set(code.toLowerCase().match(/[a-z0-9]+/g) || []);

    const hints = kb.name_hints.map(h => {
        const hintText = `${h.logic_anchor} ${h.suggested_name}`.toLowerCase();
        const hintWords = hintText.match(/[a-z0-9]+/g) || [];
        let score = 0;
        hintWords.forEach(w => {
            if (w.length > 3 && codeWords.has(w)) score++;
        });
        return { ...h, score };
    });

    const relevantHints = hints
        .filter(h => h.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, maxHints);

    if (relevantHints.length === 0) return 'None (No relevant hints found for this chunk)';

    return relevantHints.map(h => `- Logic: "${h.logic_anchor}" -> Suggested Name: "${h.suggested_name}"`).join('\n');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- MAIN STAGES ---
async function run() {
    let version = process.argv.filter((arg, i, arr) => !arg.startsWith('-') && (i === 0 || arr[i - 1] !== '--limit'))[2];
    const isRenameOnly = process.argv.includes('--rename-only') || process.argv.includes('-r');
    const isForce = process.argv.includes('--force') || process.argv.includes('-f');
    const skipVendor = process.argv.includes('--skip-vendor');
    const limitArgIdx = process.argv.indexOf('--limit');
    const limit = limitArgIdx !== -1 ? parseInt(process.argv[limitArgIdx + 1]) : Infinity;

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

    let globalMapping = {
        version: "1.2",
        variables: {},
        properties: {},
        processed_chunks: [],
        metadata: {
            total_renamed: 0,
            last_updated: new Date().toISOString()
        }
    };

    if (fs.existsSync(mappingPath)) {
        try {
            const raw = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
            if (raw.version && raw.variables) {
                globalMapping = raw;
                if (!globalMapping.processed_chunks) globalMapping.processed_chunks = [];
                console.log(`[*] Loaded structured mapping v${globalMapping.version} with ${Object.keys(globalMapping.variables).length} variables, ${Object.keys(globalMapping.properties).length} properties, and ${globalMapping.processed_chunks.length} processed chunks.`);
            } else {
                // Migration from flat format
                console.log(`[*] Migrating old flat mapping to structure format...`);
                for (const [key, val] of Object.entries(raw)) {
                    globalMapping.variables[key] = {
                        name: val,
                        confidence: 0.8,
                        source: "migration"
                    };
                }
                console.log(`[*] Migration complete: ${Object.keys(globalMapping.variables).length} entries.`);
                fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
            }
        } catch (err) {
            console.warn(`[!] Error reading mapping.json, starting fresh: ${err.message}`);
        }
    }

    const graphMapPath = path.join(versionPath, 'metadata', 'graph_map.json');
    let graphData = [];
    if (fs.existsSync(graphMapPath)) {
        try {
            const rawData = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
            graphData = rawData.chunks || rawData; // Handle both old and new formats
        } catch (err) {
            console.error(`[!] Error parsing graph_map.json: ${err.message}`);
            process.exit(1);
        }
    } else {
        console.error(`[!] Error: graph_map.json not found at ${graphMapPath}`);
        process.exit(1);
    }

    // Sort by Centrality (descending) AND Category priority
    const categoryPriority = { 'founder': 3, 'family': 2, 'vendor': 1, 'unknown': 0 };
    let sortedChunks = graphData.slice().sort((a, b) => {
        const primary = (b.centrality || 0) - (a.centrality || 0);
        if (Math.abs(primary) > 0.0001) return primary;

        // Tie-breaker: Category
        const catA = categoryPriority[a.category] || 0;
        const catB = categoryPriority[b.category] || 0;
        return catB - catA;
    });

    if (limit < sortedChunks.length) {
        console.log(`[*] Limit applied: Only processing top ${limit} chunks.`);
        sortedChunks = sortedChunks.slice(0, limit);
    }

    console.log(`[*] Starting Deobfuscation Pipeline [Provider: ${PROVIDER}, Model: ${MODEL}]`);
    console.log(`[*] Processing ${sortedChunks.length} chunks.`);

    // --- KEY VALIDATION ---
    const isValid = await validateKey();
    if (!isValid) {
        process.exit(1);
    }

    // Stage 1: Incremental Mapping
    if (isRenameOnly) {
        console.log(`[*] Skipping Stage 1 (LLM Pass) as --rename-only is set.`);
    } else {
        for (let i = 0; i < sortedChunks.length; i++) {
            const chunkMeta = sortedChunks[i];
            const file = path.basename(chunkMeta.file);
            const chunkPath = path.join(chunksDir, file);

            if (!fs.existsSync(chunkPath)) {
                console.warn(`[!] Skip: Chunk file not found: ${chunkPath}`);
                continue;
            }

            // Skip vendor logic
            if (skipVendor && chunkMeta.category === 'vendor') {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                    globalMapping.processed_chunks.push(chunkMeta.name);
                    fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
                }
                console.log(`[*] Stage 1 [${i + 1}/${sortedChunks.length}]: Skip ${file} (Vendor - skipped by flag).`);
                continue;
            }

            // Resume check: Skip if already processed according to tracking list
            if (globalMapping.processed_chunks.includes(chunkMeta.name) && !isForce) {
                console.log(`[*] Stage 1 [${i + 1}/${sortedChunks.length}]: Skip ${file} (Already processed).`);
                continue;
            }

            const code = fs.readFileSync(chunkPath, 'utf8');
            const { variables, properties } = extractIdentifiers(code);

            // Candidate Pool: Determine which identifiers REALLY need naming
            const unknownVariables = variables.filter(v => !globalMapping.variables[v]);
            const unknownProperties = properties.filter(p => !globalMapping.properties[p]);

            if (unknownVariables.length === 0 && unknownProperties.length === 0) {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                    globalMapping.processed_chunks.push(chunkMeta.name);
                    fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
                }
                console.log(`[*] Stage 1 [${i + 1}/${sortedChunks.length}]: Skip ${file} (All identifiers mapped).`);
                continue;
            }

            const filteredMapping = { variables: {}, properties: {} };
            let varMatches = 0;
            let propMatches = 0;

            for (const key of variables) {
                if (globalMapping.variables[key]) {
                    filteredMapping.variables[key] = globalMapping.variables[key].name;
                    varMatches++;
                }
            }
            for (const key of properties) {
                if (globalMapping.properties[key]) {
                    filteredMapping.properties[key] = globalMapping.properties[key].name;
                    propMatches++;
                }
            }

            // Prepare neighbor information with display names or roles
            const neighborInfo = chunkMeta.outbound.map(targetId => {
                const target = graphData.find(c => c.name === targetId || path.basename(c.file, '.js') === targetId);
                if (target) {
                    const name = target.displayName || target.suggestedFilename || target.role || targetId;
                    return `${targetId} (${name})`;
                }
                return targetId;
            }).join(', ');

            console.log(`[*] Stage 1 [${i + 1}/${sortedChunks.length}]: Naming Pass for ${file} (${chunkMeta.centrality.toFixed(4)} centrality)...`);
            console.log(`    - Context: ${varMatches} variables, ${propMatches} properties injected from mapping.`);
            console.log(`    - Discovery: ${unknownVariables.length} variables and ${unknownProperties.length} properties need naming.`);

            const prompt = `
Role: Senior Reverse Engineer
Task: Semantic Identity Mapping (Unobfuscation)

Analyze the provided JavaScript code chunk from an AI Agent CLI tool.
Goal: Map obfuscated variables and object properties to human-readable names.

CHUNK METADATA:
- Role: ${chunkMeta.role}
- Label: ${chunkMeta.label}
- Logical Source: ${chunkMeta.kb_info?.suggested_path || 'unknown'}
- Bundle Line Range: ${chunkMeta.startLine} - ${chunkMeta.endLine}
- State DNA: This code interacts with global state properties: ${chunkMeta.state_touchpoints.join(', ')}
- Neighbors: Interacts with ${neighborInfo}

UNKNOWN IDENTIFIERS TO MAP:
- Variables: ${unknownVariables.join(', ')}
- Properties: ${unknownProperties.join(', ')}

EXISTING MAPPINGS (Use these for consistency):
${JSON.stringify(filteredMapping, null, 2)}

REFERENCE HINTS FROM KNOWLEDGE BASE:
${filterKBHints(code, KB)}

CODE:
\`\`\`javascript
${code}
\`\`\`

INSTRUCTIONS:
1. Identify the purpose of obfuscated variables AND object properties based on their usage.
2. CATEGORIZE your suggestions into "variables" (let, const, var, function names) and "properties" (this.prop, obj.prop, {prop: ...}).
3. For "variables", prioritize names that reflect the variable's role in the ${chunkMeta.role} logic.
4. For "properties", ensure names reflect the data being stored or the action being performed.
5. If the logic matches a REFERENCE HINT, you MUST use that suggested name.
6. Suggest a concise, descriptive FILENAME for this chunk (e.g., 'anthropicApiClient').

CRITICAL:
- I have provided a list of UNKNOWN IDENTIFIERS. Focus your response primarily on mapping these.
- DO NOT propose new names for identifiers already present in the EXISTING MAPPINGS list; use them for consistency.
- Your response must be valid JSON and ONLY JSON.

RESPONSE FORMAT (JSON ONLY):
{
  "mappings": {
    "variables": {
       "obfuscatedVar": { "name": "descriptiveVar", "rationale": "...", "confidence": 0.9 }
    },
    "properties": {
       "obfuscatedProp": { "name": "descriptiveProp", "rationale": "...", "confidence": 0.9 }
    }
  },
  "suggestedFilename": "descriptive_name",
  "overallRationale": "Summary of choosing these names."
}
`;

            try {
                const llmResponse = await callLLM(prompt);
                const jsonMatch = llmResponse.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    const responseData = JSON.parse(jsonMatch[0]);
                    const newMappings = responseData.mappings || {};

                    let added = 0;
                    // Update variables
                    if (newMappings.variables) {
                        for (const [key, mapping] of Object.entries(newMappings.variables)) {
                            const newName = typeof mapping === 'string' ? mapping : mapping.name;
                            if (!globalMapping.variables[key] || globalMapping.variables[key].name !== newName) {
                                globalMapping.variables[key] = typeof mapping === 'string' ? { name: mapping, confidence: 0.8, source: file } : { ...mapping, source: file };
                                added++;
                            }
                        }
                    }

                    // Update properties
                    if (newMappings.properties) {
                        for (const [key, mapping] of Object.entries(newMappings.properties)) {
                            const newName = typeof mapping === 'string' ? mapping : mapping.name;
                            if (!globalMapping.properties[key] || globalMapping.properties[key].name !== newName) {
                                globalMapping.properties[key] = typeof mapping === 'string' ? { name: mapping, confidence: 0.8, source: file } : { ...mapping, source: file };
                                added++;
                            }
                        }
                    }

                    if (added > 0 || responseData.suggestedFilename) {
                        if (added > 0) {
                            globalMapping.metadata.total_renamed = Object.keys(globalMapping.variables).length + Object.keys(globalMapping.properties).length;
                            globalMapping.metadata.last_updated = new Date().toISOString();
                        }

                        // Always mark as processed if we got a valid JSON response
                        if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                            globalMapping.processed_chunks.push(chunkMeta.name);
                        }

                        fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));

                        if (responseData.suggestedFilename) {
                            chunkMeta.suggestedFilename = responseData.suggestedFilename;
                            fs.writeFileSync(graphMapPath, JSON.stringify(graphData, null, 2));
                        }

                        console.log(`    - Added/Updated ${added} mappings. [File Suggestion: ${responseData.suggestedFilename || 'None'}] Rationale: ${responseData.overallRationale || responseData.rationale || 'None'}`);
                    }

                    // Throttle to strictly abide by 10 RPM free tier limits (6s baseline + 1s jitter)
                    const throttleDelay = 6000 + Math.random() * 1000;
                    await sleep(throttleDelay);
                } else {
                    console.warn(`    [!] Could not parse JSON from LLM response for ${file}`);
                }
            } catch (err) {
                console.error(`    [!] Critical error processing ${file}: ${err.message}`);
                // We continue to the next chunk instead of crashing the whole process
            }
        }

        console.log(`[*] Stage 1 Complete. Final mapping preserved at ${mappingPath}`);
    }

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
