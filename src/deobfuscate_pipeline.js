require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

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
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx', 'typescript']
        });

        const variables = new Set();
        const properties = new Set();
        const keywords = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'enum', 'await', 'async', 'null', 'true', 'false', 'undefined']);
        const globals = new Set(['console', 'Object', 'Array', 'String', 'Number', 'Boolean', 'Promise', 'Error', 'JSON', 'Math', 'RegExp', 'Map', 'Set', 'WeakMap', 'WeakSet', 'globalThis', 'window', 'global', 'process', 'require', 'module', 'exports', 'URL', 'Buffer']);

        const builtInProps = new Set(['toString', 'constructor', 'hasOwnProperty', 'valueOf', 'propertyIsEnumerable', 'toLocaleString', 'isPrototypeOf', '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__', '__proto__']);

        traverse(ast, {
            Identifier(path) {
                const id = path.node.name;
                if (keywords.has(id) || globals.has(id) || builtInProps.has(id)) return;

                // Check if it's a property of a member expression (not computed)
                if (path.parentPath.isMemberExpression({ property: path.node, computed: false })) {
                    if (id.length > 1) properties.add(id);
                    return;
                }

                // Check if it's a key in an object property (not computed)
                if (path.parentPath.isObjectProperty({ key: path.node, computed: false })) {
                    if (id.length > 1) properties.add(id);
                    return;
                }

                // Otherwise, it's a variable/binding
                // IGNORE human-readable names: length > 4 or containing underscores or camelCase
                const isHumanReadable = id.length > 4 || id.includes('_') || (/[a-z]/.test(id) && /[A-Z]/.test(id));
                if (!isHumanReadable) {
                    variables.add(id);
                }
            }
        });

        return {
            variables: Array.from(variables),
            properties: Array.from(properties)
        };
    } catch (err) {
        console.warn(`[!] Babel extraction failed, falling back to regex: ${err.message}`);
        // Fallback to regex if parsing fails (e.g. invalid snippet)
        const variables = new Set();
        const properties = new Set();
        const idRegex = /\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g;
        let match;
        while ((match = idRegex.exec(code)) !== null) {
            const id = match[0];
            const isHumanReadable = id.length > 4 || id.includes('_') || (/[a-z]/.test(id) && /[A-Z]/.test(id));
            if (!isHumanReadable) variables.add(id);
        }
        return { variables: Array.from(variables), properties: [] };
    }
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

function cleanLLMResponse(text) {
    if (!text) return null;

    // 1. Remove markdown code blocks if present
    let cleaned = text.trim();
    const jsonBlockRegex = /```(?:json)?\s*([\s\S]*?)```/i;
    const match = cleaned.match(jsonBlockRegex);
    if (match) {
        cleaned = match[1].trim();
    }

    // 2. Sometimes the LLM adds text before or after the JSON.
    // We try to find the start and end of the JSON object.
    const startIdx = cleaned.indexOf('{');
    const endIdx = cleaned.lastIndexOf('}');

    if (startIdx === -1) return null;

    if (endIdx !== -1 && endIdx > startIdx) {
        cleaned = cleaned.substring(startIdx, endIdx + 1);
    } else {
        // Potentially truncated JSON
        cleaned = cleaned.substring(startIdx);
    }

    return cleaned;
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
                if (!globalMapping.metadata) {
                    globalMapping.metadata = {
                        total_renamed: 0,
                        last_updated: new Date().toISOString()
                    };
                }
                console.log(`[*] Loaded structured mapping v${globalMapping.version} with ${Object.keys(globalMapping.variables).length} variables, ${Object.keys(globalMapping.properties).length} properties, and ${globalMapping.processed_chunks.length} processed chunks.`);

                // Cleanup nulls from botched previous runs
                let cleanedCount = 0;
                const cleanupNulls = (obj) => {
                    for (const key in obj) {
                        if (obj[key] === null) {
                            delete obj[key];
                            cleanedCount++;
                        } else if (Array.isArray(obj[key])) {
                            const originalLen = obj[key].length;
                            obj[key] = obj[key].filter(e => e !== null);
                            cleanedCount += (originalLen - obj[key].length);
                            if (obj[key].length === 0) delete obj[key];
                        }
                    }
                };
                cleanupNulls(globalMapping.variables);
                cleanupNulls(globalMapping.properties);
                if (cleanedCount > 0) {
                    console.log(`[*] Cleaned up ${cleanedCount} null entries from mapping.`);
                }
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

    // Stage 1: Mapping Generation (LLM Pass)
    if (isRenameOnly) {
        console.log(`[*] Skipping Stage 1 (Mapping Generation) as --rename-only is set.`);
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

            // NEW: Check if the NN already provided high-confidence names for most things
            const anchoredCount = variables.filter(v => globalMapping.variables[v]?.source?.startsWith('anchored')).length;
            const anchorCoverage = variables.length > 0 ? anchoredCount / variables.length : 0;

            if (anchorCoverage > 0.8) {
                console.log(`[*] Stage 1 [${i + 1}/${sortedChunks.length}]: Skip ${file} (${(anchorCoverage * 100).toFixed(0)}% Anchored by NN).`);
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                    globalMapping.processed_chunks.push(chunkMeta.name);
                    fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
                }
                continue;
            }

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

            // --- CONTEXT INJECTION (GRAPH-INFORMED) ---
            const neighborChunks = [...new Set([...chunkMeta.outbound, ...graphData.filter(c => c.outbound.includes(chunkMeta.name)).map(c => c.name)])];

            const neighborMapping = { variables: {}, properties: {} };
            for (const neighborId of neighborChunks) {
                const neighborFileName = graphData.find(c => c.name === neighborId)?.file;
                if (!neighborFileName) continue;

                // Collect variables from globalMapping that originated from this neighbor
                for (const [id, entry] of Object.entries(globalMapping.variables)) {
                    if (!entry) continue;
                    const match = Array.isArray(entry) ? entry.find(e => e && e.source === path.basename(neighborFileName)) : (entry.source === path.basename(neighborFileName) ? entry : null);
                    if (match) neighborMapping.variables[id] = match.name;
                }
                for (const [id, entry] of Object.entries(globalMapping.properties)) {
                    if (!entry) continue;
                    const match = Array.isArray(entry) ? entry.find(e => e && e.source === path.basename(neighborFileName)) : (entry.source === path.basename(neighborFileName) ? entry : null);
                    if (match) neighborMapping.properties[id] = match.name;
                }
            }

            const filteredMapping = {
                variables: { ...neighborMapping.variables },
                properties: { ...neighborMapping.properties }
            };

            // Prioritize current chunk's IDs that are already mapped
            for (const key of variables) {
                if (globalMapping.variables[key]) {
                    const entry = globalMapping.variables[key];
                    filteredMapping.variables[key] = Array.isArray(entry) ? entry[0].name : entry.name;
                }
            }
            for (const key of properties) {
                if (globalMapping.properties[key]) {
                    const entry = globalMapping.properties[key];
                    filteredMapping.properties[key] = Array.isArray(entry) ? entry[0].name : entry.name;
                }
            }

            const MAX_INJECTED_VARS = 150;
            const MAX_INJECTED_PROPS = 150;

            const finalFilteredMapping = {
                variables: Object.fromEntries(Object.entries(filteredMapping.variables).slice(0, MAX_INJECTED_VARS)),
                properties: Object.fromEntries(Object.entries(filteredMapping.properties).slice(0, MAX_INJECTED_PROPS))
            };

            const varCount = Object.keys(finalFilteredMapping.variables).length;
            const propCount = Object.keys(finalFilteredMapping.properties).length;

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
            console.log(`    - Context: ${varCount} variables, ${propCount} properties injected from mapping (including neighbors).`);
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

EXISTING MAPPINGS (Subset for context):
${JSON.stringify(finalFilteredMapping, null, 2)}
${(varCount > MAX_INJECTED_VARS || propCount > MAX_INJECTED_PROPS) ? `\n(Note: ${varCount} variables and ${propCount} properties matched, showing only top ${MAX_INJECTED_VARS}/${MAX_INJECTED_PROPS})` : ''}

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
- KEEP rationale and overallRationale extremely concise (one sentence each) to avoid hitting token limits for large chunks.

RESPONSE FORMAT (JSON ONLY):
{
  "mappings": {
    "variables": {
       "obfuscatedVar": { "name": "descriptiveVar", "rationale": "Shorter is better.", "confidence": 0.9 }
    },
    "properties": {
       "obfuscatedProp": { "name": "descriptiveProp", "rationale": "Shorter is better.", "confidence": 0.9 }
    }
  },
  "suggestedFilename": "descriptive_name",
  "overallRationale": "Concise summary."
}
`;

            const MAX_RETRIES = 2;
            let llmProcessed = false;

            for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
                try {
                    const llmResponse = await callLLM(prompt);
                    const cleanedJson = cleanLLMResponse(llmResponse);

                    if (!cleanedJson) {
                        throw new Error("No JSON found in LLM response");
                    }

                    let responseData;
                    try {
                        const { jsonrepair } = require('jsonrepair');
                        responseData = JSON.parse(jsonrepair(cleanedJson));
                    } catch (parseErr) {
                        // Fallback to manual fix if jsonrepair fails or isn't perfect for this truncation
                        const openBraces = (cleanedJson.match(/\{/g) || []).length;
                        const closeBraces = (cleanedJson.match(/\}/g) || []).length;
                        if (openBraces > closeBraces) {
                            let fixedJson = cleanedJson.trim();
                            // If it ends with something that looks like it's inside a string (no quote)
                            if (fixedJson.match(/[a-zA-Z0-9_\-]$/)) {
                                fixedJson += '"'; // Close the string
                            }
                            // If it ends with a quote but no comma/brace, it might be a property value
                            if (fixedJson.endsWith('"')) {
                                fixedJson += ''; // No-op, just a marker
                            }

                            fixedJson += '}'.repeat(openBraces - closeBraces);
                            try {
                                responseData = JSON.parse(fixedJson);
                                console.log(`    [*] Attempted manual truncation fix was successful.`);
                            } catch (e2) {
                                // One more try: maybe it needs a comma before the closing brace if it's a field
                                try {
                                    const altFix = cleanedJson.trim() + '"}'.repeat(openBraces - closeBraces);
                                    responseData = JSON.parse(altFix);
                                    console.log(`    [*] Attempted manual truncation fix (alt) was successful.`);
                                } catch (e3) {
                                    throw new Error(`JSON parse error even after repair attempts: ${parseErr.message}`);
                                }
                            }
                        } else {
                            throw new Error(`JSON parse error: ${parseErr.message}`);
                        }
                    }

                    const newMappings = responseData.mappings || {};
                    let added = 0;

                    // Update variables
                    if (newMappings.variables) {
                        for (const [key, mapping] of Object.entries(newMappings.variables)) {
                            const newMapping = typeof mapping === 'string' ? { name: mapping, confidence: 0.8, source: chunkMeta.name } : { ...mapping, source: chunkMeta.name };
                            const existing = globalMapping.variables[key];

                            if (!existing) {
                                globalMapping.variables[key] = newMapping;
                                added++;
                            } else if (Array.isArray(existing)) {
                                const sameSourceIdx = existing.findIndex(e => e && e.source === file);
                                if (sameSourceIdx !== -1) {
                                    existing[sameSourceIdx] = newMapping;
                                } else {
                                    existing.push(newMapping);
                                }
                                added++;
                            } else if (existing && existing.source !== file) {
                                globalMapping.variables[key] = [existing, newMapping];
                                added++;
                            } else if (existing) {
                                if (existing.name !== newMapping.name || newMapping.confidence > existing.confidence) {
                                    globalMapping.variables[key] = newMapping;
                                    added++;
                                }
                            } else {
                                // Fallback just in case existing was null
                                globalMapping.variables[key] = newMapping;
                                added++;
                            }
                        }
                    }

                    // Update properties
                    if (newMappings.properties) {
                        for (const [key, mapping] of Object.entries(newMappings.properties)) {
                            const newMapping = typeof mapping === 'string' ? { name: mapping, confidence: 0.8, source: chunkMeta.name } : { ...mapping, source: chunkMeta.name };
                            const existing = globalMapping.properties[key];

                            if (!existing) {
                                globalMapping.properties[key] = newMapping;
                                added++;
                            } else if (Array.isArray(existing)) {
                                const sameSourceIdx = existing.findIndex(e => e && e.source === file);
                                if (sameSourceIdx !== -1) {
                                    existing[sameSourceIdx] = newMapping;
                                } else {
                                    existing.push(newMapping);
                                }
                                added++;
                            } else if (existing && existing.source !== file) {
                                globalMapping.properties[key] = [existing, newMapping];
                                added++;
                            } else if (existing) {
                                if (existing.name !== newMapping.name || newMapping.confidence > existing.confidence) {
                                    globalMapping.properties[key] = newMapping;
                                    added++;
                                }
                            } else {
                                // Fallback just in case existing was null
                                globalMapping.properties[key] = newMapping;
                                added++;
                            }
                        }
                    }

                    if (added > 0 || responseData.suggestedFilename) {
                        if (added > 0) {
                            globalMapping.metadata.total_renamed = Object.keys(globalMapping.variables).length + Object.keys(globalMapping.properties).length;
                            globalMapping.metadata.last_updated = new Date().toISOString();
                        }

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

                    llmProcessed = true;
                    // Throttle to strictly abide by 10 RPM free tier limits
                    const throttleDelay = 6000 + Math.random() * 1000;
                    await sleep(throttleDelay);
                    break; // Success! Exit retry loop.

                } catch (err) {
                    console.warn(`    [!] Error processing ${file} (Attempt ${attempt + 1}/${MAX_RETRIES + 1}): ${err.message}`);
                    if (attempt < MAX_RETRIES) {
                        const waitTime = 5000 * (attempt + 1);
                        console.log(`    [*] Retrying in ${waitTime / 1000}s...`);
                        await sleep(waitTime);
                    } else {
                        console.error(`    [!] Failed to process ${file} after ${MAX_RETRIES + 1} attempts. Skipping.`);
                    }
                }
            }
        }

        console.log(`[*] Stage 1 Complete. Final mapping preserved at ${mappingPath}`);
    }

    // Stage 2: Mapping Application & Renaming (Babel)
    console.log(`[*] Starting Stage 2: Applying Mapping & Renaming with Babel...`);
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
