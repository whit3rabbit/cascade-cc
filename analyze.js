const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const { encode } = require('gpt-tokenizer');
const { webcrack } = require('webcrack');
const { AAdecode, jjdecode } = require('./deobfuscators');
const t = require('@babel/types');

// --- 1. CONFIGURATION ---
const PACKAGE_NAME = '@anthropic-ai/claude-code';
const ANALYSIS_DIR = './claude-analysis';
const OUTPUT_ROOT = './cascade_graph_analysis';
let OUTPUT_BASE = OUTPUT_ROOT; // Will be updated with version
const SIGNAL_KEYWORDS = ['anthropic', 'claude', 'mcp', 'agent', 'terminal', 'prompt', 'session', 'protocol', 'codeloop'];
const NATIVE_PROPS = ['toString', 'hasOwnProperty', 'constructor', 'prototype', 'call', 'apply', 'bind'];
const GLOBAL_VARS = ['console', 'window', 'document', 'process', 'module', 'require', 'exports', 'global', 'Buffer', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval'];

function ensureTargetExists() {
    let targetFile = null;
    let version = 'unknown';

    // 1. Check command line argument
    if (process.argv[2]) {
        targetFile = process.argv[2];
        // Try to infer version from path
        const match = targetFile.match(/claude-analysis\/([^/]+)\/cli\.js/);
        if (match) version = match[1];
        return { targetFile, version };
    }

    // 2. Fetch latest version from NPM
    console.log(`[*] Checking latest version of ${PACKAGE_NAME}...`);
    try {
        version = execSync(`npm view ${PACKAGE_NAME} version`).toString().trim();
    } catch (err) {
        console.warn(`[!] Failed to fetch NPM version: ${err.message}`);
        // Fallback to locally existing if any
        if (fs.existsSync(ANALYSIS_DIR)) {
            const versions = fs.readdirSync(ANALYSIS_DIR).filter(v => /^\d+\.\d+\.\d+/.test(v)).sort().reverse();
            for (const v of versions) {
                const potentialPath = path.join(ANALYSIS_DIR, v, 'cli.js');
                if (fs.existsSync(potentialPath)) {
                    console.log(`[*] Using locally found version: ${v} (NPM check failed)`);
                    return { targetFile: potentialPath, version: v };
                }
            }
        }
    }

    // 3. Check if versioned folder already exists
    if (version !== 'unknown') {
        const versionPath = path.join(ANALYSIS_DIR, version);
        const potentialPath = path.join(versionPath, 'cli.js');
        if (fs.existsSync(potentialPath)) {
            console.log(`[*] Version ${version} already present. Skipping download.`);
            return { targetFile: potentialPath, version };
        }
    }

    // 4. If not found or if version check succeeded but folder doesn't exist, download latest from npm
    if (version !== 'unknown') {
        console.log(`[*] Downloading version ${version} of ${PACKAGE_NAME}...`);
        try {
            const versionPath = path.join(ANALYSIS_DIR, version);
            if (!fs.existsSync(versionPath)) {
                fs.mkdirSync(versionPath, { recursive: true });
                execSync(`npm pack ${PACKAGE_NAME}@${version}`, { cwd: versionPath });
                const tarball = fs.readdirSync(versionPath).find(f => f.endsWith('.tgz'));
                execSync(`tar -xzf "${tarball}" --strip-components=1`, { cwd: versionPath });
            }

            const downloadedCli = path.join(versionPath, 'cli.js');
            if (fs.existsSync(downloadedCli)) {
                return { targetFile: downloadedCli, version };
            }
        } catch (err) {
            console.error(`[!] Failed to download bundle: ${err.message}`);
        }
    }

    return { targetFile: null, version: 'unknown' };
}

/**
 * Robustly find all strings in a node tree without requiring a full Babel path/scope.
 */
function findStrings(node, strings = []) {
    if (!node) return strings;
    if (node.type === 'StringLiteral') {
        strings.push(node.value);
    } else {
        for (const key in node) {
            const child = node[key];
            if (child && typeof child === 'object') {
                if (Array.isArray(child)) {
                    child.forEach(c => findStrings(c, strings));
                } else if (child.type) {
                    findStrings(child, strings);
                }
            }
        }
    }
    return strings;
}

// --- 2. DEOBFUSCATION HEURISTICS ---
const removeDeadCodeVisitor = {
    "IfStatement|ConditionalExpression"(path) {
        try {
            const testPath = path.get("test");
            const evaluateTest = testPath.evaluateTruthy();
            if (evaluateTest === true) {
                path.replaceWith(path.node.consequent);
            } else if (evaluateTest === false) {
                if (path.node.alternate) path.replaceWith(path.node.alternate);
                else path.remove();
            }
        } catch (e) {
            // Evaluation failed due to missing scope, skip
        }
    },
    LogicalExpression(path) {
        try {
            const { left, operator, right } = path.node;
            const leftPath = path.get("left");
            const evaluateLeft = leftPath.evaluateTruthy();

            if ((operator === "||" && evaluateLeft === true) || (operator === "&&" && evaluateLeft === false)) {
                path.replaceWith(left);
            } else if ((operator === "||" && evaluateLeft === false) || (operator === "&&" && evaluateLeft === true)) {
                path.replaceWith(right);
            }
        } catch (e) {
            // Evaluation failed, skip
        }
    },
    SequenceExpression(path) {
        const { expressions } = path.node;
        if (expressions.slice(0, -1).every(e => t.isLiteral(e))) {
            path.replaceWith(expressions[expressions.length - 1]);
        }
    }
};


class CascadeGraph {
    constructor() {
        this.nodes = new Map(); // ChunkName -> { code, tokens, category, neighbors: Set }
        this.internalNameToChunkId = new Map(); // Name -> ChunkName
        this.totalTokens = 0;
    }

    // Helper to extract names defined in a statement
    extractInternalNames(node) {
        const names = [];
        if (node.type === 'VariableDeclaration') {
            node.declarations.forEach(d => {
                if (d.id && d.id.name) names.push(d.id.name);
                if (d.id && d.id.type === 'ObjectPattern') {
                    d.id.properties.forEach(p => {
                        if (p.value && p.value.name) names.push(p.value.name);
                    });
                }
            });
        } else if (node.type === 'FunctionDeclaration' && node.id) {
            names.push(node.id.name);
        } else if (node.type === 'ClassDeclaration' && node.id) {
            names.push(node.id.name);
        }
        return names;
    }

    // Step A: Initial Pass - Group logical chunks
    identifyNodes(ast) {
        console.log(`[*] Phase 1: Identifying Chunks...`);
        const statements = ast.program.body;
        let currentChunkNodes = [];
        let currentTokens = 0;
        let currentCategory = 'unknown';
        let chunkIndex = 1;

        const finalizeChunk = () => {
            if (currentChunkNodes.length === 0) return;
            const chunkName = `chunk${chunkIndex.toString().padStart(3, '0')}`;

            // Intelligent Naming Logic
            let suggestedName = null;
            const collectedStrings = [];
            currentChunkNodes.forEach(node => {
                // Apply DCR to this node and its children (silently skip if it fails)
                try { traverse(node, { ...removeDeadCodeVisitor, noScope: true }); } catch (e) { }
                findStrings(node, collectedStrings);
            });

            for (const val of collectedStrings) {
                // Heuristics for a "good" module name
                const isPath = val.includes('/') || val.includes('\\');
                const isModule = val.includes('@') || val.endsWith('.js') || val.endsWith('.ts');
                const isClaudeSignal = val.startsWith('tengu_') || val.startsWith('claude_') || val.toLowerCase().includes('anthropic');

                if (isPath || isModule || isClaudeSignal) {
                    if (!suggestedName || val.length < suggestedName.length) {
                        if (val.length > 3 && val.length < 60) suggestedName = val;
                    }
                }
            }

            const code = currentChunkNodes.map(n => generate(n, { minified: false }).code).join('\n\n');
            const tokens = encode(code).length;

            this.nodes.set(chunkName, {
                id: chunkIndex,
                name: chunkName,
                code,
                tokens,
                neighbors: new Set(),
                score: 0,
                category: currentCategory
            });

            if (suggestedName) {
                const sanitized = suggestedName.replace(/[^a-zA-Z0-9]/g, '_').replace(/^_+|_+$/g, '');
                if (sanitized.length > 3) {
                    const nodeData = this.nodes.get(chunkName);
                    nodeData.displayName = sanitized;
                }
            }

            currentChunkNodes.forEach(node => {
                this.extractInternalNames(node).forEach(name => {
                    this.internalNameToChunkId.set(name, chunkName);
                });
            });

            chunkIndex++;
            currentChunkNodes = [];
            currentTokens = 0;
            currentCategory = 'unknown';
        };

        let currentChunkHasContent = false;

        // Dynamically detect module wrappers (common in esbuild)
        const wrapperNames = new Set();
        statements.slice(0, 150).forEach(node => {
            if (node.type === 'VariableDeclaration') {
                node.declarations.forEach(decl => {
                    const init = decl.init;
                    if (init && (init.type === 'ArrowFunctionExpression' || init.type === 'FunctionExpression')) {
                        const code = generate(decl, { minified: true }).code;
                        // Detect w = (A, Q) => () => (A && (Q = A(A = 0)), Q)
                        // Detect U = (A, Q) => () => (Q || A((Q = { exports: {} }).exports, Q), Q.exports)
                        if (code.includes('exports:{}') || code.includes('A=0') || code.includes('exports: {}')) {
                            wrapperNames.add(decl.id.name);
                        }
                    }
                });
            }
        });
        if (wrapperNames.size > 0) console.log(`[*] Detected module wrappers: ${Array.from(wrapperNames).join(', ')}`);

        statements.forEach((node) => {
            const nodeCode = generate(node, { minified: true }).code;
            const nodeTokens = encode(nodeCode).length;
            const hasSignal = SIGNAL_KEYWORDS.some(kw => nodeCode.toLowerCase().includes(kw));
            const nodeCategory = hasSignal ? 'priority' : 'vendor';

            // Check if it's an import or a module wrapper call
            const isImport = node.type === 'ImportDeclaration' ||
                (node.type === 'VariableDeclaration' && nodeCode.includes('require('));

            const isModuleWrapperCall = node.type === 'VariableDeclaration' &&
                node.declarations.length === 1 &&
                node.declarations[0].init &&
                node.declarations[0].init.type === 'CallExpression' &&
                wrapperNames.has(node.declarations[0].init.callee.name);

            const isBanal = node.type === 'ExpressionStatement' &&
                node.expression.type === 'StringLiteral' &&
                (node.expression.value === 'use strict' || node.expression.value.startsWith('__esModule'));

            // JSimplifier-inspired: Detect library/utility patterns
            const isUtility = node.type === 'VariableDeclaration' &&
                node.declarations.some(d => d.id.name && (NATIVE_PROPS.includes(d.id.name) || d.id.name.startsWith('_')));

            // Heuristic Split Conditions
            const shouldSplit = currentChunkNodes.length > 0 && (
                (isImport && currentChunkHasContent) ||
                (isModuleWrapperCall && currentTokens > 3000) || // Slightly tighter for modules
                (nodeCategory === 'priority' && currentCategory === 'vendor' && currentTokens > 1500) ||
                (currentTokens + nodeTokens > 8000) || // Lower threshold for better granularity
                (isUtility && currentTokens > 2000)
            );

            if (shouldSplit) {
                finalizeChunk();
                currentChunkHasContent = false;
            }

            currentChunkNodes.push(node);
            currentTokens += nodeTokens;

            if (!isImport && !isBanal) {
                currentChunkHasContent = true;
            }

            // Upgrade category if priority signal found
            if (nodeCategory === 'priority') currentCategory = 'priority';
            else if (currentCategory === 'unknown') currentCategory = 'vendor';
        });

        finalizeChunk();
    }

    // Step B: Neighbor Detection (The Edges)
    detectNeighbors() {
        console.log(`[*] Phase 2: Neighbor Detection (Building Edges)...`);
        const internalNames = Array.from(this.internalNameToChunkId.keys());

        for (const [chunkName, nodeData] of this.nodes) {
            internalNames.forEach(definedName => {
                const targetChunk = this.internalNameToChunkId.get(definedName);
                if (chunkName === targetChunk) return;

                // Check if this chunk references a name defined in another chunk
                // We use a simple regex-like search but check for word boundaries to avoid partial matches
                if (nodeData.code.includes(definedName)) {
                    // Quick check for word boundary if possible, else just include
                    const regex = new RegExp(`\\b${definedName}\\b`);
                    if (regex.test(nodeData.code)) {
                        nodeData.neighbors.add(targetChunk);
                    }
                }
            });
        }
    }

    // Step C: Markov Chain Analysis
    applyMarkovAnalysis() {
        console.log(`[*] Phase 3: Markov Chain Centrality Calculation...`);
        const names = Array.from(this.nodes.keys());
        const size = names.length;
        if (size === 0) return;

        let scores = new Array(size).fill(1 / size);

        for (let iter = 0; iter < 10; iter++) {
            let nextScores = new Array(size).fill(0);
            for (let i = 0; i < size; i++) {
                const node = this.nodes.get(names[i]);
                const outDegree = node.neighbors.size;

                if (outDegree > 0) {
                    const contribution = scores[i] / outDegree;
                    node.neighbors.forEach(neighborName => {
                        const neighborIdx = names.indexOf(neighborName);
                        if (neighborIdx !== -1) nextScores[neighborIdx] += contribution;
                    });
                } else {
                    for (let j = 0; j < size; j++) nextScores[j] += scores[i] / size;
                }
            }
            scores = nextScores;
        }

        names.forEach((name, i) => {
            this.nodes.get(name).score = scores[i];
        });
    }

    // Phase 4: Re-classify based on aggregated chunk data
    classify() {
        console.log(`[*] Phase 4: Final Classification...`);
        // Categories are already partially set in Phase 1, but we can refine
        // based on score or other metrics if needed.
    }

    saveResults(versionDir) {
        const chunksDir = path.join(versionDir, 'chunks');
        const metadataDir = path.join(versionDir, 'metadata');

        [chunksDir, metadataDir].forEach(d => {
            if (fs.existsSync(d)) {
                // Clear old results (shouldn't happen with new versioning, but for safety)
                fs.readdirSync(d).forEach(f => fs.unlinkSync(path.join(d, f)));
            }
            fs.mkdirSync(d, { recursive: true });
        });

        const metadata = [];

        for (const [name, node] of this.nodes) {
            const fileName = node.displayName ? `${name}_${node.displayName}.js` : `${name}.js`;
            fs.writeFileSync(path.join(chunksDir, fileName), node.code);

            metadata.push({
                name: node.displayName ? `${name} (${node.displayName})` : node.name,
                file: `chunks/${fileName}`,
                tokens: node.tokens,
                category: node.category,
                centrality: node.score,
                neighborCount: node.neighbors.size,
                outbound: Array.from(node.neighbors)
            });
        }

        fs.writeFileSync(
            path.join(metadataDir, 'graph_map.json'),
            JSON.stringify(metadata, null, 2)
        );

        fs.writeFileSync(
            path.join(metadataDir, 'graph_map.js'),
            `window.GRAPH_DATA = ${JSON.stringify(metadata, null, 2)};`
        );
    }
}

// --- 3. EXECUTION ---
async function run() {
    const { targetFile, version } = ensureTargetExists();

    if (!targetFile || !fs.existsSync(targetFile)) {
        console.error(`[!] Error: No input file found.`);
        console.error(`    Checked: CLI arguments, ./cli.js, and ./claude-analysis/*/cli.js`);
        console.error(`    Usage: node analyze.js [path-to-bundle.js]`);
        process.exit(1);
    }

    console.log(`[*] Target identified: ${targetFile}`);
    let code = fs.readFileSync(targetFile, 'utf8');

    // Phase 0: Preprocessing with Webcrack
    console.log(`[*] Phase 0: Preprocessing with Webcrack...`);
    try {
        const result = await webcrack(code);
        code = result.code;
        console.log(`[*] Preprocessing complete. Code cleaned and unminified.`);
    } catch (err) {
        console.error(`[!] Webcrack failed: ${err.message}. Proceeding with raw code.`);
    }

    let ast = parser.parse(code, {
        sourceType: 'unambiguous',
        plugins: ['jsx', 'typescript'],
        errorRecovery: true
    });

    const graph = new CascadeGraph();
    graph.identifyNodes(ast);
    graph.detectNeighbors();
    graph.applyMarkovAnalysis();
    graph.classify();

    OUTPUT_BASE = path.join(OUTPUT_ROOT, version);
    console.log(`[*] Output directory: ${OUTPUT_BASE}`);

    graph.saveResults(OUTPUT_BASE);

    console.log(`\n[COMPLETE]`);
    console.log(`Analysis saved to: ${OUTPUT_BASE}`);
    return { version, path: OUTPUT_BASE };
}

run().catch(console.error);