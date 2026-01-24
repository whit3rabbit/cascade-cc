require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const { renameIdentifiers: liveRenamer } = require('./rename_chunks');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

const KEYWORDS = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'enum', 'await', 'async', 'null', 'true', 'false', 'undefined']);
const GLOBALS = new Set(['console', 'Object', 'Array', 'String', 'Number', 'Boolean', 'Promise', 'Error', 'JSON', 'Math', 'RegExp', 'Map', 'Set', 'WeakMap', 'WeakSet', 'globalThis', 'window', 'global', 'process', 'require', 'module', 'exports', 'URL', 'Buffer']);
const BUILTIN_PROPS = new Set([
    'toString', 'constructor', 'hasOwnProperty', 'valueOf', 'propertyIsEnumerable', 'toLocaleString', 'isPrototypeOf', '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__', '__proto__',
    'length', 'map', 'forEach', 'filter', 'reduce', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'join', 'split',
    'includes', 'indexOf', 'lastIndexOf', 'apply', 'call', 'bind',
    'message', 'stack', 'name', 'code', 'status', 'headers', 'body',
    'write', 'end', 'on', 'once', 'emit', 'removeListener', 'removeAllListeners',
    'substring', 'substr', 'replace', 'trim', 'toLowerCase', 'toUpperCase', 'charAt',
    'match', 'search', 'concat', 'entries', 'keys', 'values', 'from',
    'stdout', 'stderr', 'stdin', 'destroyed', 'preInit'
]);

const IDENTIFIER_REGEX = /\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g;

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

function isChunkyPath(candidatePath, chunkNames = []) {
    if (!candidatePath || typeof candidatePath !== 'string') return false;
    const base = path.basename(candidatePath, path.extname(candidatePath)).toLowerCase();
    if (/^chunk\d+$/.test(base)) return true;
    if (/chunk\d+/i.test(base)) return true;
    if (chunkNames.some(n => base === String(n).toLowerCase())) return true;
    return false;
}

function isGenericProposedPath(candidatePath, chunkMeta) {
    if (!candidatePath || typeof candidatePath !== 'string') return true;
    const normalized = candidatePath.replace(/\\/g, '/');
    const base = path.basename(normalized, path.extname(normalized)).toLowerCase();
    const chunkName = String(chunkMeta?.name || '').toLowerCase();
    if (isChunkyPath(candidatePath, chunkName ? [chunkName] : [])) return true;
    if (normalized.includes('/core/logic/')) return true;
    return false;
}

function buildGlobalIdentifierFrequency(chunksDir, chunkFiles) {
    const freq = new Map();
    let totalChunks = 0;
    for (const file of chunkFiles) {
        const fullPath = path.join(chunksDir, file);
        if (!fs.existsSync(fullPath)) continue;
        totalChunks++;
        const code = fs.readFileSync(fullPath, 'utf8');
        const seen = new Set();
        IDENTIFIER_REGEX.lastIndex = 0;
        let match;
        while ((match = IDENTIFIER_REGEX.exec(code)) !== null) {
            const id = match[0];
            if (KEYWORDS.has(id) || GLOBALS.has(id) || BUILTIN_PROPS.has(id)) continue;
            seen.add(id);
        }
        seen.forEach(id => {
            freq.set(id, (freq.get(id) || 0) + 1);
        });
    }
    return { freq, totalChunks };
}

function buildIdentifierImportance(code) {
    const importance = new Map();
    const mark = (name, level) => {
        if (!name) return;
        const prev = importance.get(name) || 0;
        if (level > prev) importance.set(name, level);
    };

    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx', 'typescript']
        });

        traverse(ast, {
            ExportNamedDeclaration(path) {
                const decl = path.node.declaration;
                if (decl) {
                    if (decl.id && decl.id.name) mark(decl.id.name, 3);
                    if (decl.declarations) {
                        decl.declarations.forEach(d => {
                            if (d.id && d.id.name) mark(d.id.name, 3);
                        });
                    }
                }
                if (path.node.specifiers) {
                    path.node.specifiers.forEach(s => {
                        if (s.exported && s.exported.name) mark(s.exported.name, 3);
                        if (s.local && s.local.name) mark(s.local.name, 3);
                    });
                }
            },
            ExportDefaultDeclaration(path) {
                const decl = path.node.declaration;
                if (decl && decl.id && decl.id.name) mark(decl.id.name, 3);
                if (decl && decl.type === 'Identifier') mark(decl.name, 3);
            },
            FunctionDeclaration(path) {
                if (path.node.id && path.node.id.name) mark(path.node.id.name, 3);
            },
            ClassDeclaration(path) {
                if (path.node.id && path.node.id.name) mark(path.node.id.name, 3);
            },
            VariableDeclarator(path) {
                if (path.node.id && path.node.id.type === 'Identifier') {
                    const init = path.node.init;
                    if (init && (init.type === 'FunctionExpression' || init.type === 'ArrowFunctionExpression' || init.type === 'ClassExpression')) {
                        mark(path.node.id.name, 3);
                    }
                }
            },
            ReturnStatement(path) {
                if (!path.node.argument) return;
                path.traverse({
                    Identifier(p) {
                        mark(p.node.name, 2);
                    }
                });
            },
            ForStatement(path) {
                const init = path.node.init;
                if (init && init.type === 'VariableDeclaration') {
                    init.declarations.forEach(d => {
                        if (d.id && d.id.type === 'Identifier') mark(d.id.name, 1);
                    });
                }
            },
            ForInStatement(path) {
                const left = path.node.left;
                if (left && left.type === 'VariableDeclaration') {
                    left.declarations.forEach(d => {
                        if (d.id && d.id.type === 'Identifier') mark(d.id.name, 1);
                    });
                } else if (left && left.type === 'Identifier') {
                    mark(left.name, 1);
                }
            },
            ForOfStatement(path) {
                const left = path.node.left;
                if (left && left.type === 'VariableDeclaration') {
                    left.declarations.forEach(d => {
                        if (d.id && d.id.type === 'Identifier') mark(d.id.name, 1);
                    });
                } else if (left && left.type === 'Identifier') {
                    mark(left.name, 1);
                }
            }
        });
    } catch (err) {
        return importance;
    }

    return importance;
}

function extractIdentifiers(code, options = {}) {
    try {
        const { globalFreq = null, globalThreshold = null } = options;
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx', 'typescript']
        });

        const variables = new Set();
        const properties = new Set();
        const variableCandidates = new Set();
        const variableCounts = new Map();
        const isDictObfuscated = id => globalFreq && globalThreshold && (globalFreq.get(id) || 0) >= globalThreshold;

        traverse(ast, {
            Identifier(path) {
                const id = path.node.name;
                if (KEYWORDS.has(id) || GLOBALS.has(id) || BUILTIN_PROPS.has(id)) return;

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
                variableCandidates.add(id);
                variableCounts.set(id, (variableCounts.get(id) || 0) + 1);
            }
        });

        const isHumanReadable = id =>
            id.length > 4 || id.includes('_') || (/[a-z]/.test(id) && /[A-Z]/.test(id));
        const isLikelyObfuscated = (id, count) => count >= 20;

        variableCandidates.forEach(id => {
            const count = variableCounts.get(id) || 0;
            if (!isHumanReadable(id) || isLikelyObfuscated(id, count) || isDictObfuscated(id)) {
                variables.add(id);
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
        const { globalFreq = null, globalThreshold = null } = options;
        const isDictObfuscated = id => globalFreq && globalThreshold && (globalFreq.get(id) || 0) >= globalThreshold;
        IDENTIFIER_REGEX.lastIndex = 0;
        let match;
        const counts = new Map();
        while ((match = IDENTIFIER_REGEX.exec(code)) !== null) {
            const id = match[0];
            if (KEYWORDS.has(id) || GLOBALS.has(id) || BUILTIN_PROPS.has(id)) continue;
            counts.set(id, (counts.get(id) || 0) + 1);
        }
        const isHumanReadable = id =>
            id.length > 4 || id.includes('_') || (/[a-z]/.test(id) && /[A-Z]/.test(id));
        const isLikelyObfuscated = (id, count) => count >= 20;
        counts.forEach((count, id) => {
            if (!isHumanReadable(id) || isLikelyObfuscated(id, count) || isDictObfuscated(id)) variables.add(id);
        });
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

function resolveVectors(entry) {
    if (!entry) return { structural: null, literals: null };
    if (entry.vector_structural && entry.vector_literals) {
        return { structural: entry.vector_structural, literals: entry.vector_literals };
    }
    if (entry.vector) {
        return { structural: entry.vector, literals: null };
    }
    return { structural: null, literals: null };
}

function calculateWeightedSimilarity(aStruct, aLit, bStruct, bLit, structWeight = 0.7, litWeight = 0.3) {
    if (!aStruct || !bStruct) return 0;
    const structSim = calculateSimilarity(aStruct, bStruct);
    if (aLit && bLit) {
        const litSim = calculateSimilarity(aLit, bLit);
        return (structWeight * structSim) + (litWeight * litSim);
    }
    return structSim;
}

function findBestRegistryMatch(chunkVectors) {
    if (!LOGIC_REGISTRY || !chunkVectors || !chunkVectors.structural) return { label: null, ref: null, similarity: -1 };
    let bestMatch = { label: null, ref: null, similarity: -1 };
    for (const [label, refData] of Object.entries(LOGIC_REGISTRY)) {
        const refVectors = resolveVectors(refData);
        const sim = calculateWeightedSimilarity(
            chunkVectors.structural,
            chunkVectors.literals,
            refVectors.structural,
            refVectors.literals
        );
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
    const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js'));

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
    const { freq: globalIdentifierFreq, totalChunks: globalIdentifierTotal } = buildGlobalIdentifierFrequency(chunksDir, chunkFiles);
    const globalIdentifierThreshold = Math.max(5, Math.floor(globalIdentifierTotal * 0.2));

    // --- CONSOLIDATION PASS (Phase 0.9) ---
    // Group chunks by Module ID or Affinity Link to propose unified file paths *before* individual processing
    const consolidationGroups = [];
    const processedInConsolidation = new Set();

    // Ensure chronological order for consolidation grouping
    const chronoChunks = graphData.slice().sort((a, b) => { // Fixed: Ensure clean numeric sort
        const idA = parseInt(a.name.replace('chunk', ''), 10);
        const idB = parseInt(b.name.replace('chunk', ''), 10);
        return idA - idB;
    });

    for (let i = 0; i < chronoChunks.length; i++) {
        const chunk = chronoChunks[i];
        if (processedInConsolidation.has(chunk.name)) continue;

        const group = [chunk];
        processedInConsolidation.add(chunk.name);

        // Look ahead for linked chunks
        let nextIdx = i + 1;
        while (nextIdx < chronoChunks.length) {
            const next = chronoChunks[nextIdx];
            const prev = group[group.length - 1];

            const isHardLinked = (prev.moduleId && next.moduleId && prev.moduleId === next.moduleId);
            const isSoftLinked = (prev.affinityLink === next.name);

            if (isHardLinked || isSoftLinked) {
                group.push(next);
                processedInConsolidation.add(next.name);
                nextIdx++;
            } else {
                break;
            }
        }

        if (group.length > 1) {
            consolidationGroups.push(group);
        }
    }

    if (consolidationGroups.length > 0 && !isDryRun) {
        console.log(`[*] Phase 0.9: Running Consolidation Pass on ${consolidationGroups.length} groups...`);
        const pLimit = require('p-limit');
        const limit = pLimit(5); // Fast parallel check

        const runConsolidation = (group) => limit(async () => {
            // Skip if we already have good paths for all
            if (group.every(c => c.proposedPath || c.suggestedPath)) return;

            const groupMeta = group.map(c => {
                let code = c.code;
                if (!code) {
                    try {
                        const file = c.name.endsWith('.js') ? c.name : `${c.name}.js`;
                        code = fs.readFileSync(path.join(chunksDir, file), 'utf8');
                    } catch (e) {
                        code = "";
                    }
                }
                const neighbors = (c.outbound || []).slice(0, 5).map(n => {
                    const neighborMeta = graphData.find(m => m.name === n);
                    return {
                        name: neighborMeta?.displayName || neighborMeta?.name || n,
                        path: neighborMeta?.suggestedPath || neighborMeta?.proposedPath || neighborMeta?.kb_info?.suggested_path || null
                    };
                });
                return {
                    name: c.name,
                    preview: code ? code.slice(0, 300).replace(/\n/g, ' ') : "",
                    vars: (code.match(/var\s+([a-zA-Z0-9_$]+)/g) || []).slice(0, 5).join(', '),
                    role: c.role || null,
                    category: c.category || null,
                    kbPath: c.kb_info?.suggested_path || null,
                    parentHintPath: c.parentHintPath || null,
                    neighbors
                };
            });

            const prompt = `
Role: Senior Code Architect
Task: Reconstruct Original Source File from Split Chunks

We have detected that the following ${group.length} chunks belong to the SAME logical module (linked by internal closure state or affinity).
Your job is to identify the singular ORIGINAL source file they belong to.

Chunks:
${JSON.stringify(groupMeta, null, 2)}

Existing Hints:
${group.map(c => c.kb_info ? `- ${c.name}: KB suggests ${c.kb_info.suggested_path}` : '').join('\n')}

Project Structure Reference:
${KB && KB.project_structure ? JSON.stringify(KB.project_structure, null, 2) : 'Structure not available.'}

Instructions:
1. Analyze the variable scope continuity and logic.
2. Propose a single unified file path (e.g. "src/services/sessionManager.ts") that contains all these chunks.
3. Assign "part" numbers to each chunk (1, 2, 3...).
4. Do NOT use chunk IDs (e.g. "chunk002") in the filename.
5. Prefer directories hinted by parentHintPath, neighbor paths, or KB suggestions. If uncertain, use a descriptive filename based on semantics, not chunk IDs.

Response JSON:
{
  "unifiedPath": "src/path/to/file.ts",
  "rationale": "Why these chunks are one file",
  "parts": {
    "chunk001": 1,
    "chunk002": 2
  }
}
`;
            try {
                // Short timeout, this should be quick intuition
                const response = await callLLM(prompt);
                const { cleaned } = cleanLLMResponse(response);
                if (cleaned) {
                    const result = JSON.parse(cleaned);
                    if (result.unifiedPath) {
                        const groupNames = group.map(c => c.name);
                        if (isChunkyPath(result.unifiedPath, groupNames)) {
                            console.warn(`    [!] Consolidation path looks generic: ${result.unifiedPath}. Skipping proposedPath.`);
                            return;
                        }
                        console.log(`    [+] Consolidated ${group.length} chunks into ${result.unifiedPath}`);
                        group.forEach(c => {
                            c.proposedPath = result.unifiedPath;
                            c.partIndex = result.parts?.[c.name] || 1;
                            // Propagate to global mapping logic if needed
                        });
                    }
                }
            } catch (e) {
                console.warn(`    [!] Consolidation failed for group ${group[0].name}: ${e.message}`);
            }
        });

        await Promise.all(consolidationGroups.map(runConsolidation));
    }


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

        const runChunk = async (chunkMeta, precedingContext = null) => {
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

            const originalCode = fs.readFileSync(chunkPath, 'utf8');

            const needsPathInference = !chunkMeta.suggestedPath && (!chunkMeta.proposedPath || isGenericProposedPath(chunkMeta.proposedPath, chunkMeta));

            if (globalMapping.processed_chunks.includes(chunkMeta.name) && !isForce && !needsPathInference) {
                // Return existing deobfuscated code as context if available
                const deobfuscatedPath = path.join(versionPath, 'deobfuscated_chunks', file);
                try {
                    // Try to find if a renamed version exists
                    const candidates = fs.readdirSync(path.join(versionPath, 'deobfuscated_chunks')).filter(f => f.startsWith(chunkMeta.name));
                    if (candidates.length > 0) {
                        return fs.readFileSync(path.join(versionPath, 'deobfuscated_chunks', candidates[0]), 'utf8');
                    }
                } catch (e) { }

                skipProcessedCount++;
                return originalCode; // Fallback to original code for context
            }

            const { variables: origVars, properties: origProps } = extractIdentifiers(originalCode, {
                globalFreq: globalIdentifierFreq,
                globalThreshold: globalIdentifierThreshold
            });
            const importanceMap = buildIdentifierImportance(originalCode);

            // Filter for unknown identifiers using ORIGINAL mangled names
            // Filter for unknown identifiers using ORIGINAL mangled names
            // Send to LLM if: 1. No mapping exists OR 2. Mapping is low confidence OR 3. The "resolved" name is still 1-2 chars
            let unknownVariables = origVars.filter(v => {
                const m = globalMapping.variables[v];
                return !m || (m.confidence || 0) < 0.85 || (m.name && m.name.length <= 2);
            });
            let unknownProperties = origProps.filter(p => {
                const m = globalMapping.properties[p];
                return !m || (m.confidence || 0) < 0.85 || (m.name && m.name.length <= 2);
            });
            unknownVariables = unknownVariables.sort((a, b) => {
                const ia = importanceMap.get(a) || 0;
                const ib = importanceMap.get(b) || 0;
                if (ia !== ib) return ib - ia;
                return a.localeCompare(b);
            });
            unknownProperties = unknownProperties.sort((a, b) => a.localeCompare(b));

            if (unknownVariables.length === 0 && unknownProperties.length === 0 && !isForce && !needsPathInference) {
                if (!globalMapping.processed_chunks.includes(chunkMeta.name)) {
                    console.log(`    [-] Skipping ${file} (no unknown identifiers)`);
                    globalMapping.processed_chunks.push(chunkMeta.name);
                }
                skipNoUnknownCount++;
                return;
            }

            if (unknownVariables.length === 0 && unknownProperties.length === 0 && needsPathInference) {
                console.log(`    [WORKING] ${file} (path-only inference)`);
            } else {
                console.log(`    [WORKING] ${file} (${unknownVariables.length} vars, ${unknownProperties.length} props unknown)`);
            }

            // Apply current mappings to the code before sending it to the LLM
            // This makes the code much more 'human-readable' for the engine.
            const neighbors = [...(chunkMeta.neighbors || []), ...(chunkMeta.outbound || [])];
            let contextCode = originalCode;
            try {
                const partiallyRenamedCode = liveRenamer(originalCode, globalMapping, {
                    sourceFile: chunkMeta.name,
                    neighbors,
                    displayName: chunkMeta.displayName,
                    suggestedPath: chunkMeta.proposedPath || chunkMeta.kb_info?.suggested_path,
                    moduleId: chunkMeta.moduleId || null
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
            const chunkVectors = resolveVectors(logicMatch);
            const registryMatch = findBestRegistryMatch(chunkVectors);
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

            const existingPaths = sortedChunks
                .filter(c => c.suggestedPath && globalMapping.processed_chunks.includes(c.name))
                .sort((a, b) => (b.centrality || 0) - (a.centrality || 0))
                .slice(0, 20)
                .map(c => `- ${c.name} (${c.role}) -> ${c.suggestedPath}`);

            const generatePrompt = (vars, props, codeContent, goldReferenceCode = '', goldSimilarity = null, precedingContext = null) => `
Role: Staff Software Engineer (Reverse Engineering Team)
Task: Reconstruct Proprietary "Founder" Logic and File Structure

CONTEXT:
This chunk has been identified as ${chunkMeta.role}.
It is intended to be located at: ${chunkMeta.proposedPath || chunkMeta.kb_info?.suggested_path || (chunkMeta.parentHintPath ? path.join(chunkMeta.parentHintPath, 'child_module.ts') : 'src/undetermined/')}.

PROJECT STRUCTURE REFERENCE:
Use this structure to guide your filename proposals. Place files in the most appropriate directory based on their logic.
${KB && KB.project_structure ? JSON.stringify(KB.project_structure, null, 2) : 'Structure not available.'}

INFERRED FILESYSTEM STATE (Active Context):
These are the directory paths we have already confirmed for key modules. Use these to maintain consistency (e.g. if you see a neighbor listed here, put this file close to it).
${existingPaths.length > 0 ? existingPaths.join('\n') : 'No paths confirmed yet.'}

NEIGHBOR CONTEXT:
This code interacts with:
${(chunkMeta.outbound || []).map(n => {
                const neighborMeta = graphData.find(m => m.name === n);
                const pathHint = neighborMeta?.suggestedPath || neighborMeta?.proposedPath ? ` (Located: ${neighborMeta.suggestedPath || neighborMeta.proposedPath})` : '';
                return `- ${neighborMeta?.displayName || neighborMeta?.name || n}${pathHint}`;
            }).join('\n')}

NEIGHBOR ANCHOR HINT:
${globalMapping.neighbor_hints && globalMapping.neighbor_hints[chunkMeta.name]
                ? `High-confidence neighbor suggests library context: ${globalMapping.neighbor_hints[chunkMeta.name].lib} (Source: ${globalMapping.neighbor_hints[chunkMeta.name].source}, Similarity: ${(globalMapping.neighbor_hints[chunkMeta.name].similarity * 100).toFixed(2)}%)`
                : 'None'}

MAPPING KNOWLEDGE (High Confidence or Established Guesses):
The following symbols have already been identified. Use these names in your reasoning.
${[...origVars, ...origProps].filter(id => {
                const m = globalMapping.variables[id] || globalMapping.properties[id];
                return m && (m.confidence >= 0.8 || m.source.includes('bootstrap'));
            }).map(id => {
                const m = globalMapping.variables[id] || globalMapping.properties[id];
                return `- ${id} is ${m.name} (Source: ${m.source}, Confidence: ${m.confidence})`;
            }).join('\n') || 'None'}

TARGET IDENTIFIERS TO RESOLVE (prioritize higher-importance names first):
Variables: ${vars.length > 0 ? vars.join(', ') : 'None'}
Properties: ${props.length > 0 ? props.join(', ') : 'None'}

            ${goldReferenceCode ? `GOLD REFERENCE MATCH:
            This chunk matches a library signature (${(goldSimilarity * 100).toFixed(2)}% similarity). Use this to resolve ambiguous names.
            
            \`\`\`javascript
            ${goldReferenceCode}
            \`\`\`
            ` : ''}

            INHERITED MAPPINGS (From Previous Chunks in ${chunkMeta.proposedPath || 'Module'}):
            ${(() => {
                    if (chunkMeta.parentChunk || (chunkMeta.partIndex && chunkMeta.partIndex > 1)) {
                        // Find likely parent
                        const parentName = chunkMeta.parentChunk ||
                            (chunkMeta.partIndex ? graphData.find(c => c.proposedPath === chunkMeta.proposedPath && c.partIndex === chunkMeta.partIndex - 1)?.name : null);

                        if (parentName) {
                            const parentVars = globalMapping.variables;
                            const inherited = Object.entries(parentVars)
                                .filter(([k, v]) => v.source === parentName)
                                .map(([k, v]) => `- ${k} -> ${v.name} (Inherited)`)
                                .join('\n');
                            return inherited || "None relevant.";
                        }
                    }
                    return "None.";
                })()}

            ${precedingContext ? `PRECEDING CODE CONTEXT (The code immediately before this chunk in the same file):
            \`\`\`javascript
            ${precedingContext.slice(-2000)}
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
4. PROPOSE A FULL FILE PATH:
   - Priority 1: If a neighbor is already in a specific folder (see NEIGHBOR CONTEXT), prefer placing this file in the same directory or a sub-directory unless it crosses an architectural boundary.
   - Priority 2: Use the PROJECT STRUCTURE REFERENCE.
   - Example: "src/utils/stringHelpers.js" NOT just "stringHelpers"
   - If the intended path looks generic (e.g. "src/core/logic/chunk123.ts"), override it with a more descriptive path.
   - Never use chunk IDs (e.g. "chunk002") in the filename.
5. Output valid JSON only.

RESPONSE FORMAT (JSON ONLY):
{
  "mappings": {
    "variables": { "mangled": { "name": "clean"${skipRationale ? '' : ', "rationale": "..."'}, "confidence": 0.9 } },
    "properties": { "mangled": { "name": "clean"${skipRationale ? '' : ', "rationale": "..."'}, "confidence": 0.9 } }
  },
  "corrections": {
    "mangled": { "name": "new_correct_name", "rationale": "Why the previous automated mapping was wrong (e.g. results in Date.filter which is not a function)" }
  },
  "suggestedPath": "src/path/to/logical_name.js"
}
`;

            if (isDryRun) {
                console.log(`[DRY RUN] Prompt for ${file} would be generated.`);
                if (existingPaths.length > 0) console.log(`[DRY RUN] Inferred State Example: ${existingPaths[0]}`);
                return;
            }

            // Auto-splitting logic
            const VAR_CHUNK_SIZE = 40;
            const PROP_CHUNK_SIZE = 40;
            const forcePathOnly = needsPathInference && unknownVariables.length === 0 && unknownProperties.length === 0;

            // Parallel processing loop (fix for nested loop bug)
            const varBatches = forcePathOnly ? 1 : Math.ceil(unknownVariables.length / VAR_CHUNK_SIZE);
            const propBatches = forcePathOnly ? 1 : Math.ceil(unknownProperties.length / PROP_CHUNK_SIZE);
            const maxBatches = forcePathOnly ? 1 : Math.max(varBatches, propBatches);

            for (let i = 0; i < maxBatches; i++) {
                const vOffset = i * VAR_CHUNK_SIZE;
                const pOffset = i * PROP_CHUNK_SIZE;

                const varSub = forcePathOnly ? [] : unknownVariables.slice(vOffset, vOffset + VAR_CHUNK_SIZE);
                const propSub = forcePathOnly ? [] : unknownProperties.slice(pOffset, pOffset + PROP_CHUNK_SIZE);

                if (!forcePathOnly && varSub.length === 0 && propSub.length === 0) continue;

                const isFirstBatch = forcePathOnly ? true : (vOffset === 0 && pOffset === 0);
                const codeToPass = isFirstBatch ? contextCode : skeletonize(contextCode);
                const prompt = generatePrompt(varSub, propSub, codeToPass, goldReferenceCode, goldSimilarity, precedingContext);
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
                        if (responseData.suggestedPath) {
                            if (isChunkyPath(responseData.suggestedPath, [chunkMeta.name])) {
                                console.warn(`    [!] Ignoring generic suggestedPath for ${chunkMeta.name}: ${responseData.suggestedPath}`);
                            } else {
                                chunkMeta.suggestedPath = responseData.suggestedPath;
                                // Also populate suggestedFilename for backward compatibility
                                chunkMeta.suggestedFilename = path.basename(responseData.suggestedPath, path.extname(responseData.suggestedPath));

                                // Proactive Hinting: Tell neighbors about this decision
                                const neighborNames = [...(chunkMeta.neighbors || []), ...(chunkMeta.outbound || [])];
                                neighborNames.forEach(nName => {
                                    const neighbor = sortedChunks.find(m => m.name === nName); // Look up in sortedChunks which references the graph objects
                                    if (neighbor && !neighbor.suggestedPath) {
                                        neighbor.parentHintPath = path.dirname(responseData.suggestedPath);
                                    }
                                });
                            }
                        } else if (responseData.suggestedFilename) {
                            if (isChunkyPath(`${responseData.suggestedFilename}.js`, [chunkMeta.name])) {
                                console.warn(`    [!] Ignoring generic suggestedFilename for ${chunkMeta.name}: ${responseData.suggestedFilename}`);
                            } else {
                                chunkMeta.suggestedFilename = responseData.suggestedFilename;
                            }
                        }

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

            // --- INCREMENTAL SYNC ---
            // Apply current mappings and save to deobfuscated_chunks immediately
            const deobfuscatedDir = path.join(versionPath, 'deobfuscated_chunks');
            if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

            let finalRenamedCode = null;

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

                finalRenamedCode = liveRenamer(originalCode, globalMapping, {
                    sourceFile: chunkMeta.name,
                    neighbors,
                    displayName: chunkMeta.displayName,
                    suggestedPath: chunkMeta.proposedPath || chunkMeta.kb_info?.suggested_path,
                    moduleId: chunkMeta.moduleId || null
                });

                // Generate Metadata Block
                const metadataBlock = `/**
 * ------------------------------------------------------------------
 * Deobfuscated Chunk: ${chunkMeta.displayName || chunkMeta.name}
 * ------------------------------------------------------------------
 * Category: ${chunkMeta.category}
 * Role: ${chunkMeta.role}
 * Proposed Path: ${chunkMeta.proposedPath || 'N/A'}
 *
 * KB Info:
 * ${chunkMeta.kb_info ? JSON.stringify(chunkMeta.kb_info, null, 2).split('\n').map(line => ' * ' + line).join('\n') : 'None'}
 *
 * Related Chunks:
 * ${(chunkMeta.outbound || []).map(n => ` * - ${n}`).join('\n') || ' * None'}
 * ------------------------------------------------------------------
 */
`;

                const finalContent = metadataBlock + '\n' + (finalRenamedCode || originalCode);
                fs.writeFileSync(outputPath, finalContent);
            } catch (err) {
                console.warn(`    [!] Incremental sync failed for ${file}: ${err.message}`);
            }

            globalMapping.processed_chunks.push(chunkMeta.name);
            // Save progress frequently (every chunk)
            // Save progress frequently (every chunk)
            fs.writeFileSync(mappingPath, JSON.stringify(globalMapping, null, 2));

            return finalRenamedCode || originalCode;
        };

        // --- SCHEDULER: Grouped Execution ---
        const chunkToGroup = new Map();
        consolidationGroups.forEach(group => {
            group.forEach(c => chunkToGroup.set(c.name, group));
        });

        const executeBatch = async (chunks) => {
            const workUnits = [];
            const scheduled = new Set();

            for (const chunk of chunks) {
                if (scheduled.has(chunk.name)) continue;

                const group = chunkToGroup.get(chunk.name);
                if (group) {
                    workUnits.push(group);
                    group.forEach(c => scheduled.add(c.name));
                } else {
                    workUnits.push([chunk]);
                    scheduled.add(chunk.name);
                }
            }

            // Execute work units
            await Promise.all(workUnits.map(unit => limit(async () => {
                let context = null;
                for (const unitChunk of unit) {
                    // Start of unit (or first processed chunk) needs no prev context, only the chain does
                    const result = await runChunk(unitChunk, context);
                    if (result) context = result;
                }
            })));
        };

        console.log(`[*] Processing Core Batch (${coreChunks.length} chunks)...`);
        await executeBatch(coreChunks);

        console.log(`[*] Processing Remaining Batch (${otherChunks.length} chunks)...`);
        await executeBatch(otherChunks);

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
