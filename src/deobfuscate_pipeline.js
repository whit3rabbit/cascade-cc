require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const { renameIdentifiers: liveRenamer } = require('./rename_chunks');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

const OUTPUT_ROOT = './cascade_graph_analysis';
const REGISTRY_PATH = path.join(OUTPUT_ROOT, 'logic_registry.json');
const BOOTSTRAP_SOURCE_ROOT = './ml/bootstrap_data';

// --- KNOWLEDGE BASE ---
const KB_PATH = './knowledge_base.json';
let KB = null;
if (fs.existsSync(KB_PATH)) {
    KB = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
    console.log(`[*] Loaded Knowledge Base with ${KB.name_hints?.length || 0} name hints.`);
    if (KB.project_structure) {
        console.log(`[*] Loaded Project Structure Definition.`);
    }
}

// --- REFERENCE CONTEXT ---
const REFERENCE_ROOT = './example_output/2.1.9';
let REFERENCE_TREE = "";

function loadReferenceTree(dir, prefix = "") {
    if (!fs.existsSync(dir)) return;
    const files = fs.readdirSync(dir);
    files.forEach(file => {
        const fullPath = path.join(dir, file);
        const relPath = path.relative(REFERENCE_ROOT, fullPath);
        if (fs.statSync(fullPath).isDirectory()) {
            REFERENCE_TREE += `${prefix}DIR: ${relPath}\n`;
            loadReferenceTree(fullPath, prefix + "  ");
        } else if (file.endsWith('.js') || file.endsWith('.jsx')) {
            REFERENCE_TREE += `${prefix}FILE: ${relPath}\n`;
        }
    });
}

if (fs.existsSync(REFERENCE_ROOT)) {
    console.log(`[*] Loading Reference Structure from ${REFERENCE_ROOT}...`);
    loadReferenceTree(REFERENCE_ROOT);
}

// --- LOGIC REGISTRY (GOLD REFERENCES) ---
let LOGIC_REGISTRY = null;
if (fs.existsSync(REGISTRY_PATH)) {
    LOGIC_REGISTRY = JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
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

        const builtInProps = new Set([
            'toString', 'constructor', 'hasOwnProperty', 'valueOf', 'propertyIsEnumerable', 'toLocaleString', 'isPrototypeOf', '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__', '__proto__',
            'length', 'map', 'forEach', 'filter', 'reduce', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'join', 'split',
            'includes', 'indexOf', 'lastIndexOf', 'apply', 'call', 'bind',
            'message', 'stack', 'name', 'code', 'status', 'headers', 'body',
            'write', 'end', 'on', 'once', 'emit', 'removeListener', 'removeAllListeners',
            'substring', 'substr', 'replace', 'trim', 'toLowerCase', 'toUpperCase', 'charAt',
            'match', 'search', 'concat', 'entries', 'keys', 'values', 'from',
            'stdout', 'stderr', 'stdin', 'destroyed', 'preInit'
        ]);

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

function skeletonize(code) {
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx', 'typescript']
        });

        traverse(ast, {
            StringLiteral(path) {
                path.node.value = '';
            },
            NumericLiteral(path) {
                path.node.value = 0;
            },
            TemplateLiteral(path) {
                path.node.quasis = [{ type: 'TemplateElement', value: { raw: '', cooked: '' }, tail: true }];
                path.node.expressions = [];
            }
        });

        const { code: skeleton } = generate(ast, { compact: true, minified: true, comments: false });
        return skeleton;
    } catch (err) {
        // Fallback: simple regex to strip strings if Babel fails
        return code.replace(/(['"])(?:(?!\1|\\).|\\.)*\1/g, '""').substring(0, 5000);
    }
}

const { generate } = require('@babel/generator').default || require('@babel/generator');

// --- KB OPTIMIZATION ---
let KB_HINTS_MAP = null;
function initKBHintsMap(kb) {
    if (!kb || !kb.name_hints) return;
    KB_HINTS_MAP = new Map();
    kb.name_hints.forEach(hint => {
        const words = `${hint.logic_anchor} ${hint.suggested_name}`.toLowerCase().match(/[a-z0-9]+/g) || [];
        words.forEach(w => {
            if (w.length > 3) {
                if (!KB_HINTS_MAP.has(w)) KB_HINTS_MAP.set(w, new Set());
                KB_HINTS_MAP.get(w).add(hint);
            }
        });
    });
}

function filterKBHints(code, kb, chunkVector = null, logicDb = [], maxHints = 100) {
    if (!kb || !kb.name_hints) return 'None';
    if (!KB_HINTS_MAP) initKBHintsMap(kb);

    const codeWords = new Set(code.toLowerCase().match(/[a-z0-9]+/g) || []);
    const candidateHints = new Map(); // hint -> score

    // Word-based scoring
    codeWords.forEach(w => {
        const hints = KB_HINTS_MAP.get(w);
        if (hints) {
            hints.forEach(h => {
                candidateHints.set(h, (candidateHints.get(h) || 0) + 1);
            });
        }
    });

    // Vector-based scoring (if available)
    if (chunkVector && logicDb.length > 0) {
        const calculateSim = (v1, v2) => v1.reduce((sum, a, i) => sum + a * v2[i], 0);
        kb.name_hints.forEach(hint => {
            if (hint.logic_vector) {
                const sim = calculateSim(chunkVector, hint.logic_vector);
                if (sim > 0.9) {
                    candidateHints.set(hint, (candidateHints.get(hint) || 0) + sim * 10);
                }
            }
        });
    }

    const relevantHints = Array.from(candidateHints.keys())
        .map(h => ({ ...h, score: candidateHints.get(h) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, maxHints);

    if (relevantHints.length === 0) return 'None (No relevant hints found for this chunk)';

    return relevantHints.map(h => `- Logic: "${h.logic_anchor}" -> Suggested Name: "${h.suggested_name}"`).join('\n');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function cleanLLMResponse(text) {
    if (!text) return { cleaned: null, isTruncated: false };

    // 1. Try to find JSON in markdown blocks first
    const jsonBlockRegex = /```(?:json)?\s*([\s\S]*?)```/gi;
    let match;
    while ((match = jsonBlockRegex.exec(text)) !== null) {
        const potential = match[1].trim();
        try {
            // Simple validation, jsonrepair will do the heavy lifting
            if (potential.startsWith('{') && potential.endsWith('}')) {
                return { cleaned: potential, isTruncated: false };
            }
        } catch (e) { }
    }

    // 2. Fallback: Find the first '{' and last '}'
    let cleaned = text.trim();
    const startIdx = cleaned.indexOf('{');
    const endIdx = cleaned.lastIndexOf('}');

    if (startIdx === -1) return { cleaned: null, isTruncated: false };

    const isTruncated = endIdx === -1 || (cleaned.length - endIdx > 20 && !cleaned.substring(endIdx).toLowerCase().includes('}'));

    if (endIdx !== -1 && endIdx > startIdx) {
        cleaned = cleaned.substring(startIdx, endIdx + 1);
    } else {
        cleaned = cleaned.substring(startIdx);
    }

    return { cleaned, isTruncated };
}

function calculateSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    return vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
}

function findBestRegistryMatch(chunkVector) {
    if (!LOGIC_REGISTRY || !chunkVector) return { label: null, ref: null, similarity: -1 };
    let bestMatch = { label: null, ref: null, similarity: -1 };
    for (const [label, refData] of Object.entries(LOGIC_REGISTRY)) {
        const sim = calculateSimilarity(chunkVector, refData.vector);
        if (sim > bestMatch.similarity) {
            bestMatch = { label, ref: refData, similarity: sim };
        }
    }
    return bestMatch;
}

function loadGoldSource(match) {
    if (!match || !match.ref || !match.label) return '';
    const bootstrapRoot = path.join(OUTPUT_ROOT, 'bootstrap');
    if (!fs.existsSync(bootstrapRoot)) return '';

    let libName = match.ref.lib || null;
    let chunkName = match.ref.chunk_name || null;

    if (!libName || !chunkName) {
        const lastUnderscore = match.label.lastIndexOf('_');
        if (lastUnderscore !== -1) {
            libName = match.label.slice(0, lastUnderscore);
            chunkName = match.label.slice(lastUnderscore + 1);
        }
    }

    if (!libName || !chunkName) return '';
    const chunksDir = path.join(bootstrapRoot, libName, 'chunks');
    if (!fs.existsSync(chunksDir)) return '';

    const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js'));
    const preferred = `${chunkName}.js`;
    const matchFile = chunkFiles.find(f => f === preferred) ||
        chunkFiles.find(f => f.startsWith(`${chunkName}_`));
    if (!matchFile) return '';

    const goldPath = path.join(chunksDir, matchFile);
    const raw = fs.readFileSync(goldPath, 'utf8');
    const MAX_GOLD_REF_CHARS = 4000;
    return raw.length > MAX_GOLD_REF_CHARS ? `${raw.slice(0, MAX_GOLD_REF_CHARS)}\n// ... truncated` : raw;
}

function loadGoldSourceFromBootstrapData(label, libName = null) {
    if (!label && !libName) return '';
    if (label) {
        const directPath = path.join(BOOTSTRAP_SOURCE_ROOT, `${label}_source.js`);
        if (fs.existsSync(directPath)) {
            return fs.readFileSync(directPath, 'utf8');
        }
    }

    if (libName) {
        const bundledPath = path.join(BOOTSTRAP_SOURCE_ROOT, libName, 'bundled.js');
        if (fs.existsSync(bundledPath)) {
            return fs.readFileSync(bundledPath, 'utf8');
        }
    }

    return '';
}

function truncateReference(code) {
    const MAX_GOLD_REF_CHARS = 4000;
    if (!code) return '';
    return code.length > MAX_GOLD_REF_CHARS ? `${code.slice(0, MAX_GOLD_REF_CHARS)}\n// ... truncated` : code;
}

// --- MAIN STAGES ---
async function run() {
    let version = process.argv.filter((arg, i, arr) => !arg.startsWith('-') && (i === 0 || arr[i - 1] !== '--limit'))[2];
    const isRenameOnly = process.argv.includes('--rename-only') || process.argv.includes('-r');
    const skipRationale = process.argv.includes('--no-rationale');
    const isDryRun = process.argv.includes('--dry-run');
    const isForce = process.argv.includes('--force') || process.argv.includes('-f');
    const skipVendor = process.argv.includes('--skip-vendor');
    const limitArgIdx = process.argv.indexOf('--limit');
    let limitValue = limitArgIdx !== -1 ? parseInt(process.argv[limitArgIdx + 1]) : Infinity;

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
        console.error(`    Available versions: ${fs.readdirSync(OUTPUT_ROOT).filter(f => fs.statSync(path.join(OUTPUT_ROOT, f)).isDirectory()).map(String).join(', ') || 'None'}`);
        process.exit(1);
    }

    const chunksDir = path.join(versionPath, 'chunks');
    const mappingPath = path.join(versionPath, 'metadata', 'mapping.json');
    const graphMapPath = path.join(versionPath, 'metadata', 'graph_map.json');
    const logicDbPath = path.join(versionPath, 'metadata', 'logic_db.json');

    if (!fs.existsSync(chunksDir)) {
        console.error(`[!] Error: Chunks directory not found at ${chunksDir}`);
        process.exit(1);
    }

    let globalMapping = {
        version: "1.2",
        variables: {},
        properties: {},
        processed_chunks: [],
        metadata: { total_renamed: 0, last_updated: new Date().toISOString() }
    };

    if (fs.existsSync(mappingPath)) {
        try {
            globalMapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
            if (!globalMapping.processed_chunks) globalMapping.processed_chunks = [];
        } catch (err) { console.warn(`[!] Error reading mapping.json: ${err.message}`); }
    }

    let graphData = [];
    let logicDb = [];
    if (fs.existsSync(graphMapPath)) {
        const raw = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
        graphData = Array.isArray(raw) ? raw : (raw.chunks || []);
    }
    if (fs.existsSync(logicDbPath)) {
        logicDb = JSON.parse(fs.readFileSync(logicDbPath, 'utf8'));
    }
    const logicDbByName = new Map(logicDb.map(entry => [entry.name, entry]));

    const CORE_LIBS = ['zod', 'react', 'react-dom', 'next', 'lucide', 'framer-motion', 'clsx', 'tailwind', 'radix-ui'];

    let sortedChunks = graphData.slice().sort((a, b) => {
        // 1. Prioritize Core Libraries
        const aIsCore = CORE_LIBS.some(lib => a.displayName?.toLowerCase().includes(lib) || a.proposedPath?.toLowerCase().includes(lib));
        const bIsCore = CORE_LIBS.some(lib => b.displayName?.toLowerCase().includes(lib) || b.proposedPath?.toLowerCase().includes(lib));
        if (aIsCore && !bIsCore) return -1;
        if (!aIsCore && bIsCore) return 1;

        // 2. Prioritize Centrality (Business Logic Hubs)
        return (b.centrality || 0) - (a.centrality || 0);
    });

    if (limitValue < sortedChunks.length) sortedChunks = sortedChunks.slice(0, limitValue);

    let newMappingsCount = 0;
    let correctionCount = 0;
    let skipProcessedCount = 0;
    let skipNoUnknownCount = 0;
    let skipVendorCount = 0;

    console.log(`[*] Starting Deobfuscation Pipeline [Provider: ${PROVIDER}, Model: ${MODEL}]`);
    if (isDryRun) console.log(`[!] DRY RUN MODE: No LLM calls will be made.`);

    const isValid = await validateKey();
    if (!isValid) process.exit(1);

    if (isRenameOnly) {
        console.log(`[*] Skipping Stage 1 (Mapping Generation) as --rename-only is set.`);
    } else {
        const pLimit = require('p-limit');
        const limit = pLimit(PROVIDER === 'gemini' ? 1 : 3);

        const coreChunks = sortedChunks.filter(c => CORE_LIBS.some(lib => c.displayName?.toLowerCase().includes(lib) || c.proposedPath?.toLowerCase().includes(lib)));
        const otherChunks = sortedChunks.filter(c => !coreChunks.includes(c));

        const runChunk = (chunkMeta) => limit(async () => {
            const file = path.basename(chunkMeta.file);
            const chunkPath = path.join(chunksDir, file);
            if (!fs.existsSync(chunkPath)) return;

            if (skipVendor && chunkMeta.category === 'vendor') {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                    console.log(`    [-] Skipping ${file} (vendor)`);
                    globalMapping.processed_chunks.push(chunkMeta.name);
                }
                skipVendorCount++;
                return;
            }

            if (globalMapping.processed_chunks.includes(chunkMeta.name) && !isForce) {
                skipProcessedCount++;
                return;
            }

            const originalCode = fs.readFileSync(chunkPath, 'utf8');
            const { variables: origVars, properties: origProps } = extractIdentifiers(originalCode);

            // Filter for unknown identifiers using ORIGINAL mangled names
            const unknownVariables = origVars.filter(v => !globalMapping.variables[v] || (globalMapping.variables[v].confidence || 0) < 0.9);
            const unknownProperties = origProps.filter(p => !globalMapping.properties[p] || (globalMapping.properties[p].confidence || 0) < 0.9);

            if (unknownVariables.length === 0 && unknownProperties.length === 0 && !isForce) {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                    console.log(`    [-] Skipping ${file} (no unknown identifiers)`);
                    globalMapping.processed_chunks.push(chunkMeta.name);
                }
                skipNoUnknownCount++;
                return;
            }

            console.log(`    [WORKING] ${file} (${unknownVariables.length} vars, ${unknownProperties.length} props unknown)`);

            // Apply current mappings to the code before sending it to the LLM
            // This makes the code much more 'human-readable' for the engine.
            const neighbors = [...(chunkMeta.neighbors || []), ...(chunkMeta.outbound || [])];
            let contextCode = originalCode;
            try {
                const partiallyRenamedCode = liveRenamer(originalCode, globalMapping, {
                    sourceFile: chunkMeta.name,
                    neighbors,
                    displayName: chunkMeta.displayName,
                    suggestedPath: chunkMeta.proposedPath || chunkMeta.kb_info?.suggested_path
                });
                if (partiallyRenamedCode) {
                    contextCode = partiallyRenamedCode;
                }
            } catch (err) {
                console.warn(`    [!] Pre-renaming failed for ${file}: ${err.message}. Sending original code.`);
            }

            const logicMatch = logicDbByName.get(chunkMeta.name);
            const logicLabel = logicMatch?.bestMatchLabel || logicMatch?.label || null;
            const logicSimilarity = logicMatch?.bestMatchSimilarity || logicMatch?.similarity || null;
            const chunkVector = logicMatch?.vector;
            const registryMatch = findBestRegistryMatch(chunkVector);
            const effectiveMatch = logicLabel
                ? {
                    label: logicLabel,
                    ref: registryMatch.ref,
                    similarity: typeof logicSimilarity === 'number' ? logicSimilarity : registryMatch.similarity
                }
                : registryMatch;

            const goldSimilarity = effectiveMatch.similarity;
            let goldReferenceCode = '';
            if (goldSimilarity >= 0.95) {
                goldReferenceCode = loadGoldSourceFromBootstrapData(effectiveMatch.label, effectiveMatch.ref?.lib);
                if (!goldReferenceCode) {
                    goldReferenceCode = loadGoldSource(effectiveMatch);
                }
                goldReferenceCode = truncateReference(goldReferenceCode);
            }

            const generatePrompt = (vars, props, codeContent, goldReferenceCode = '', goldSimilarity = null) => `
Role: Staff Software Engineer (Reverse Engineering Team)
Task: Reconstruct Proprietary "Founder" Logic

CONTEXT:
This chunk has been identified as ${chunkMeta.role}.
It is intended to be located at: ${chunkMeta.proposedPath || chunkMeta.kb_info?.suggested_path || 'src/undetermined/'}.

PROJECT STRUCTURE REFERENCE:
Use this structure to guide your filename proposals. Place files in the most appropriate directory based on their logic.
${KB && KB.project_structure ? JSON.stringify(KB.project_structure, null, 2) : 'Structure not available.'}

NEIGHBOR CONTEXT:
This code interacts with:
${(chunkMeta.outbound || []).map(n => {
                const neighborMeta = graphData.find(m => m.name === n);
                return `- ${neighborMeta?.displayName || neighborMeta?.name || n}`;
            }).join('\n')}

MAPPING KNOWLEDGE (High Confidence or Established Guesses):
The following symbols have already been identified. Use these names in your reasoning.
${[...origVars, ...origProps].filter(id => {
                const m = globalMapping.variables[id] || globalMapping.properties[id];
                return m && (m.confidence >= 0.8 || m.source.includes('bootstrap'));
            }).map(id => {
                const m = globalMapping.variables[id] || globalMapping.properties[id];
                return `- ${id} is ${m.name} (Source: ${m.source}, Confidence: ${m.confidence})`;
            }).join('\n') || 'None'}

${goldReferenceCode ? `GOLD REFERENCE MATCH:
This chunk matches a library signature (${(goldSimilarity * 100).toFixed(2)}% similarity). Use this to resolve ambiguous names.

\`\`\`javascript
${goldReferenceCode}
\`\`\`
` : ''}

SOURCE CODE:
\`\`\`javascript
${codeContent}
\`\`\`

INSTRUCTIONS:
1. Examine the code for logical consistency. If an existing mapping (listed above) results in nonsensical code (e.g. \`Date.filter()\`), suggest a corrected name in the "corrections" block.
2. Identify the 'Proprietary' business logic that is unique to Claude.
3. Rename the remaining mangled variables (single letters/short names) based on their semantic usage.
4. Output valid JSON only.

RESPONSE FORMAT (JSON ONLY):
{
  "mappings": {
    "variables": { "mangled": { "name": "clean"${skipRationale ? '' : ', "rationale": "..."'}, "confidence": 0.9 } },
    "properties": { "mangled": { "name": "clean"${skipRationale ? '' : ', "rationale": "..."'}, "confidence": 0.9 } }
  },
  "corrections": {
    "mangled": { "name": "new_correct_name", "rationale": "Why the previous automated mapping was wrong (e.g. results in Date.filter which is not a function)" }
  },
  "suggestedFilename": "descriptive_name"
}
`;

            if (isDryRun) {
                console.log(`[DRY RUN] Prompt for ${file} would be generated.`);
                return;
            }

            // Auto-splitting logic
            const VAR_CHUNK_SIZE = 40;
            const PROP_CHUNK_SIZE = 40;

            for (let vOffset = 0; vOffset < unknownVariables.length; vOffset += VAR_CHUNK_SIZE) {
                for (let pOffset = 0; pOffset < unknownProperties.length; pOffset += PROP_CHUNK_SIZE) {
                    const varSub = unknownVariables.slice(vOffset, vOffset + VAR_CHUNK_SIZE);
                    const propSub = unknownProperties.slice(pOffset, pOffset + PROP_CHUNK_SIZE);
                    if (varSub.length === 0 && propSub.length === 0) continue;

                    const isFirstBatch = (vOffset === 0 && pOffset === 0);
                    const codeToPass = isFirstBatch ? contextCode : skeletonize(contextCode);
                    const prompt = generatePrompt(varSub, propSub, codeToPass, goldReferenceCode, goldSimilarity);
                    const PROMPT_RETRIES = 3;
                    let success = false;

                    for (let attempt = 1; attempt <= PROMPT_RETRIES; attempt++) {
                        try {
                            const llmResponse = await callLLM(prompt);
                            const { cleaned: cleanedJson, isTruncated } = cleanLLMResponse(llmResponse);
                            if (!cleanedJson) throw new Error("No JSON found in LLM response");

                            const { jsonrepair } = require('jsonrepair');
                            const responseData = JSON.parse(jsonrepair(cleanedJson));

                            const updateMapping = (source, target, chunkName, usedNamesInThisChunk) => {
                                for (const [key, mapping] of Object.entries(source)) {
                                    if (!mapping) continue;
                                    const newName = typeof mapping === 'string' ? mapping : mapping.name;
                                    if (!newName) continue;
                                    if (usedNamesInThisChunk.has(newName)) continue;
                                    usedNamesInThisChunk.add(newName);

                                    const newEntry = typeof mapping === 'string'
                                        ? { name: newName, confidence: 0.8, source: chunkName }
                                        : { ...mapping, name: newName, source: chunkName, confidence: mapping.confidence || 0.8 };

                                    if (target[key]) {
                                        const existing = target[key];
                                        if (newEntry.confidence > (existing.confidence || 0)) {
                                            if (existing.name !== newEntry.name) newMappingsCount++;
                                            target[key] = newEntry;
                                        }
                                    } else {
                                        target[key] = newEntry;
                                        newMappingsCount++;
                                    }
                                }
                            };

                            if (responseData.mappings?.variables) {
                                const usedNamesInThisChunk = new Set();
                                updateMapping(responseData.mappings.variables, globalMapping.variables, chunkMeta.name, usedNamesInThisChunk);
                            }
                            if (responseData.mappings?.properties) {
                                const usedNamesInThisChunk = new Set();
                                updateMapping(responseData.mappings.properties, globalMapping.properties, chunkMeta.name, usedNamesInThisChunk);
                            }

                            // HANDLE CORRECTIONS (Override even high-confidence errors)
                            if (responseData.corrections) {
                                for (const [mangled, correction] of Object.entries(responseData.corrections)) {
                                    const newName = typeof correction === 'string' ? correction : correction.name;
                                    const rationale = correction.rationale || "LLM semantic correction";
                                    console.log(`    [*] Corrected mapping for ${mangled}: -> ${newName} (${rationale})`);

                                    const targetColl = globalMapping.variables[mangled] ? globalMapping.variables : (globalMapping.properties[mangled] ? globalMapping.properties : null);
                                    if (targetColl) {
                                        targetColl[mangled] = {
                                            name: newName,
                                            confidence: 0.98, // High confidence for manual/LLM overrides
                                            source: `${chunkMeta.name}_correction`,
                                            rationale
                                        };
                                        correctionCount++;
                                    }
                                }
                            }
                            if (responseData.suggestedFilename) chunkMeta.suggestedFilename = responseData.suggestedFilename;

                            console.log(`    [+] Mapped identifiers in ${file} (Sub-pass)`);
                            success = true;
                            break; // Exit retry loop on success
                        } catch (err) {
                            const isTransient = err.message?.includes('JSON') || err.message?.includes('Colon expected') || err.message?.includes('Unexpected token');
                            if (isTransient && attempt < PROMPT_RETRIES) {
                                const backoff = Math.pow(2, attempt) * 2000 + Math.random() * 1000;
                                console.warn(`    [!] Attempt ${attempt}/${PROMPT_RETRIES} failed for ${file}: ${err.message}. Retrying in ${Math.round(backoff / 1000)}s...`);
                                await sleep(backoff);
                            } else {
                                console.warn(`    [!] Error ${file} (Attempt ${attempt}/${PROMPT_RETRIES}): ${err.message}`);
                                if (attempt === PROMPT_RETRIES) break;
                            }
                        }
                    }

                    // Reliability delay between batches
                    const { PROVIDER_CONFIG } = require('./llm_client');
                    const delay = (PROVIDER_CONFIG && PROVIDER_CONFIG[PROVIDER]?.delay) || 3000;
                    await sleep(delay + Math.random() * 1000);
                }
            }

            // --- INCREMENTAL SYNC ---
            // Apply current mappings and save to deobfuscated_chunks immediately
            const deobfuscatedDir = path.join(versionPath, 'deobfuscated_chunks');
            if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

            try {
                let logicalName = "";
                if (chunkMeta.proposedPath) {
                    logicalName = path.basename(chunkMeta.proposedPath, '.ts');
                } else if (chunkMeta.suggestedFilename) {
                    logicalName = chunkMeta.suggestedFilename;
                } else if (chunkMeta.kb_info && chunkMeta.kb_info.suggested_path) {
                    logicalName = path.basename(chunkMeta.kb_info.suggested_path.replace(/`/g, ''), '.ts').replace('.js', '');
                }
                logicalName = logicalName.replace(/[\/\\?%*:|"<>]/g, '_');

                const chunkBase = path.basename(file, '.js');
                const finalName = logicalName ? `${chunkBase}_${logicalName}.js` : file;
                const outputPath = path.join(deobfuscatedDir, finalName);

                const finalRenamedCode = liveRenamer(originalCode, globalMapping, {
                    sourceFile: chunkMeta.name,
                    neighbors,
                    displayName: chunkMeta.displayName,
                    suggestedPath: chunkMeta.proposedPath || chunkMeta.kb_info?.suggested_path
                });

                fs.writeFileSync(outputPath, finalRenamedCode || originalCode);
            } catch (err) {
                console.warn(`    [!] Incremental sync failed for ${file}: ${err.message}`);
            }

            globalMapping.processed_chunks.push(chunkMeta.name);
            // Save progress frequently (every chunk)
            fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
        });


        console.log(`[*] Processing Core Batch (${coreChunks.length} chunks)...`);
        await Promise.all(coreChunks.map(runChunk));

        console.log(`[*] Processing Remaining Batch (${otherChunks.length} chunks)...`);
        await Promise.all(otherChunks.map(runChunk));

        // Persist updated metadata back to graph_map.json
        console.log(`[*] Persisting updated metadata (suggested filenames) to graph_map.json...`);
        const originalMetadata = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
        if (Array.isArray(originalMetadata)) {
            fs.writeFileSync(graphMapPath, JSON.stringify(graphData, null, 2));
        } else if (originalMetadata.chunks) {
            originalMetadata.chunks = graphData;
            fs.writeFileSync(graphMapPath, JSON.stringify(originalMetadata, null, 2));
        }

        console.log(`\n[*] Stage 1 Complete.`);
        console.log(`    - New Mappings: ${newMappingsCount}`);
        console.log(`    - Corrections:  ${correctionCount}`);
        console.log(`    - Skipped (Prev. Processed): ${skipProcessedCount}`);
        console.log(`    - Skipped (No Unknown Idents): ${skipNoUnknownCount}`);
        console.log(`    - Skipped (Vendor): ${skipVendorCount}`);
        console.log(`    - Mapping file updated: ${path.basename(mappingPath)}`);
    }

    // Ensure metadata is synchronized if we're in a state where it might be stale
    if (isRenameOnly && !isDryRun) {
        console.log(`[*] Syncing metadata for rename-only pass...`);
        const originalMetadata = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
        if (originalMetadata.chunks && graphData.length > 0) {
            originalMetadata.chunks = graphData;
            fs.writeFileSync(graphMapPath, JSON.stringify(originalMetadata, null, 2));
        } else if (Array.isArray(originalMetadata) && graphData.length > 0) {
            fs.writeFileSync(graphMapPath, JSON.stringify(graphData, null, 2));
        }
    }

    if (!isDryRun) {
        console.log(`[*] Starting Stage 2: Applying Mapping & Renaming...`);
        execSync(`node src/rename_chunks.js "${versionPath}"`, { stdio: 'inherit' });
    }
}

run().catch(err => {
    console.error(`\n[FATAL ERROR] Pipeline aborted: ${err.message}`);
    process.exit(1);
});
