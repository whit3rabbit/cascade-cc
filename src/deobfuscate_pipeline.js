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
        console.error(`    Available versions: ${fs.readdirSync(OUTPUT_ROOT).filter(f => fs.statSync(path.join(OUTPUT_ROOT, f)).isDirectory()).join(', ') || 'None'}`);
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

    let sortedChunks = graphData.slice().sort((a, b) => (b.centrality || 0) - (a.centrality || 0));
    if (limitValue < sortedChunks.length) sortedChunks = sortedChunks.slice(0, limitValue);

    console.log(`[*] Starting Deobfuscation Pipeline [Provider: ${PROVIDER}, Model: ${MODEL}]`);
    if (isDryRun) console.log(`[!] DRY RUN MODE: No LLM calls will be made.`);

    const isValid = await validateKey();
    if (!isValid) process.exit(1);

    if (isRenameOnly) {
        console.log(`[*] Skipping Stage 1 (Mapping Generation) as --rename-only is set.`);
    } else {
        const pLimit = require('p-limit');
        const limit = pLimit(PROVIDER === 'gemini' ? 1 : 3);

        const tasks = sortedChunks.map((chunkMeta) => limit(async () => {
            const file = path.basename(chunkMeta.file);
            const chunkPath = path.join(chunksDir, file);
            if (!fs.existsSync(chunkPath)) return;

            if (skipVendor && chunkMeta.category === 'vendor') {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) globalMapping.processed_chunks.push(chunkMeta.name);
                return;
            }

            if (globalMapping.processed_chunks.includes(chunkMeta.name) && !isForce) return;

            const code = fs.readFileSync(chunkPath, 'utf8');
            const { variables, properties } = extractIdentifiers(code);

            const unknownVariables = variables.filter(v => !globalMapping.variables[v] || (globalMapping.variables[v].confidence || 0) < 0.9);
            const unknownProperties = properties.filter(p => !globalMapping.properties[p] || (globalMapping.properties[p].confidence || 0) < 0.9);

            if (unknownVariables.length === 0 && unknownProperties.length === 0 && !isForce) {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) globalMapping.processed_chunks.push(chunkMeta.name);
                return;
            }

            const generatePrompt = (vars, props, codeContent) => `
Role: Senior Reverse Engineer
Task: Semantic Identity Mapping (Unobfuscation)

Analyze the provided JavaScript code chunk.
Goal: Map obfuscated identifiers to human-readable names.

CHUNK METADATA:
- Role: ${chunkMeta.role}
- Neighbors: ${chunkMeta.outbound.join(', ')}

UNKNOWN IDENTIFIERS:
- Variables: ${vars.join(', ')}
- Properties: ${props.join(', ')}

CODE:
\`\`\`javascript
${codeContent}
\`\`\`

EXISTING HINTS (from previous stages):
${[...variables, ...properties].filter(id => globalMapping.variables[id] || globalMapping.properties[id]).map(id => {
                const m = globalMapping.variables[id] || globalMapping.properties[id];
                return `- ${id} is likely "${m.name}" (Confidence: ${m.confidence})`;
            }).join('\n') || 'None'}

RESPONSE FORMAT (JSON ONLY):
{
  "mappings": {
    "variables": { "mangled": { "name": "clean"${skipRationale ? '' : ', "rationale": "..."'}, "confidence": 0.9 } },
    "properties": { "mangled": { "name": "clean"${skipRationale ? '' : ', "rationale": "..."'}, "confidence": 0.9 } }
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
                    const codeToPass = isFirstBatch ? code : skeletonize(code);
                    const prompt = generatePrompt(varSub, propSub, codeToPass);
                    try {
                        const llmResponse = await callLLM(prompt);
                        const { cleaned: cleanedJson, isTruncated } = cleanLLMResponse(llmResponse);
                        if (!cleanedJson) throw new Error("No JSON found");

                        const { jsonrepair } = require('jsonrepair');
                        const responseData = JSON.parse(jsonrepair(cleanedJson));

                        const updateMapping = (source, target, chunkName) => {
                            for (const [key, mapping] of Object.entries(source)) {
                                if (!mapping) continue;
                                const newEntry = typeof mapping === 'string' ? { name: mapping, confidence: 0.8, source: chunkName } : { ...mapping, source: chunkName };

                                if (target[key]) {
                                    const existing = target[key];
                                    if (newEntry.confidence > (existing.confidence || 0)) {
                                        target[key] = newEntry;
                                    }
                                } else {
                                    target[key] = newEntry;
                                }
                            }
                        };

                        if (responseData.mappings?.variables) updateMapping(responseData.mappings.variables, globalMapping.variables, chunkMeta.name);
                        if (responseData.mappings?.properties) updateMapping(responseData.mappings.properties, globalMapping.properties, chunkMeta.name);
                        if (responseData.suggestedFilename) chunkMeta.suggestedFilename = responseData.suggestedFilename;

                        console.log(`    [+] Mapped identifiers in ${file} (Sub-pass)`);

                        // Reliability delay
                        const { PROVIDER_CONFIG } = require('./llm_client');
                        const delay = (PROVIDER_CONFIG && PROVIDER_CONFIG[PROVIDER]?.delay) || 3000;
                        await sleep(delay + Math.random() * 1000);
                    } catch (err) {
                        console.warn(`    [!] Error ${file}: ${err.message}`);
                    }
                }
            }

            globalMapping.processed_chunks.push(chunkMeta.name);
            // Save progress frequently (every chunk)
            fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
        }));

        await Promise.all(tasks);
        fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));
        console.log(`[*] Stage 1 Complete. Mapping saved.`);
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
