require('dotenv').config();
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const { encode } = require('gpt-tokenizer');
const { webcrack } = require('webcrack');
const { AAdecode, jjdecode } = require('./deobfuscation_helpers');
const t = require('@babel/types');

// --- 0. KNOWLEDGE BASE ---
const KB_PATH = './knowledge_base.json';
let KB = null;
if (fs.existsSync(KB_PATH)) {
    KB = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
    console.log(`[*] Loaded Knowledge Base with ${KB.file_anchors?.length || 0} file anchors.`);
}

// --- 1. CONFIGURATION ---
const PACKAGE_NAME = '@anthropic-ai/claude-code';
const ANALYSIS_DIR = './claude-analysis';
const OUTPUT_ROOT = './cascade_graph_analysis';
let OUTPUT_BASE = OUTPUT_ROOT; // Will be updated with version
const IS_BOOTSTRAP = process.argv.includes('--is-bootstrap');
const SIGNAL_KEYWORDS = ['anthropic', 'claude', 'mcp', 'agent', 'terminal', 'prompt', 'session', 'protocol', 'codeloop'];
const NATIVE_PROPS = ['toString', 'hasOwnProperty', 'constructor', 'prototype', 'call', 'apply', 'bind'];
const GLOBAL_VARS = ['console', 'window', 'document', 'process', 'module', 'require', 'exports', 'global', 'Buffer', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval'];

function ensureTargetExists() {
    let targetFile = null;
    let version = 'unknown';

    // 1. Check command line argument
    if (process.argv[2]) {
        targetFile = path.resolve(process.argv[2]);
        // Handle --version flag
        const versionIdx = process.argv.indexOf('--version');
        if (versionIdx !== -1 && process.argv[versionIdx + 1]) {
            version = process.argv[versionIdx + 1];
        } else {
            // Try to infer version from path: claude-analysis/<version>/cli.js or similar
            const match = targetFile.match(/claude-analysis\/([^/]+)\/cli\.js/) || targetFile.match(/([0-9]+\.[0-9]+\.[0-9]+)/);
            if (match) version = match[1];
        }

        // If still unknown, use a hash of the file path to avoid collisions
        if (version === 'unknown') {
            const crypto = require('crypto');
            version = 'custom-' + crypto.createHash('md5').update(targetFile).digest('hex').slice(0, 8);
        }

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
        process.stdout.write(`[*] Downloading version ${version} of ${PACKAGE_NAME}...\r`);
        try {
            const versionPath = path.join(ANALYSIS_DIR, version);
            if (!fs.existsSync(versionPath)) {
                fs.mkdirSync(versionPath, { recursive: true });
                execSync(`npm pack ${PACKAGE_NAME}@${version}`, { cwd: versionPath, stdio: 'ignore' });
                const tarball = fs.readdirSync(versionPath).find(f => f.endsWith('.tgz'));
                execSync(`tar -xzf "${tarball}" --strip-components=1`, { cwd: versionPath, stdio: 'ignore' });
            }

            // Verify version from package.json if possible
            const pkgJsonPath = path.join(versionPath, 'package.json');
            if (fs.existsSync(pkgJsonPath)) {
                const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
                if (pkg.version) version = pkg.version;
            }

            const downloadedCli = path.join(versionPath, 'cli.js');
            if (fs.existsSync(downloadedCli)) {
                return { targetFile: downloadedCli, version };
            }
        } catch (err) {
            console.error(`\n[!] Failed to download bundle: ${err.message}`);
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

// --- NEW: STRUCTURAL AST EXPORT FOR ML ---
const crypto = require('crypto');

/**
 * Strips identifiers and values from a node to create a "shape-only" representation.
 * Now includes a literal hashing channel to preserve semantic signal.
 */
function simplifyAST(node) {
    if (!node || typeof node !== 'object') return null;
    if (Array.isArray(node)) {
        return node.map((item, idx) => {
            const res = simplifyAST(item);
            if (res) res.slot = idx;
            return res;
        }).filter(Boolean);
    }

    const structure = { type: node.type };
    if (node.callee && node.callee.name) structure.call = node.callee.name;
    if (node.type === 'Identifier') structure.name = node.name;

    // Literal Hashing Channel
    if (node.type === 'StringLiteral' || node.type === 'NumericLiteral' || node.type === 'BooleanLiteral') {
        const val = String(node.value);
        structure.valHash = crypto.createHash('md5').update(val).digest('hex').slice(0, 8);
    }

    const children = [];
    for (const key in node) {
        const val = node[key];
        if (val && typeof val === 'object' && key !== 'loc' && key !== 'start' && key !== 'end' && key !== 'comments') {
            const childResult = simplifyAST(val);
            if (childResult) {
                if (Array.isArray(childResult)) {
                    childResult.forEach(c => {
                        if (c) {
                            c.slot = key;
                            children.push(c);
                        }
                    });
                } else {
                    childResult.slot = key;
                    children.push(childResult);
                }
            }
        }
    }

    if (children.length > 0) {
        structure.children = children;
    }

    return structure;
}

function collapseProxyNodes(nodes) {
    console.log(`[*] Phase 2.5: Collapsing Proxy Nodes (Metadata-Safe)...`);
    let collapsedCount = 0;

    const names = Array.from(nodes.keys());
    for (const name of names) {
        const node = nodes.get(name);
        if (!node) continue;

        // Simple heuristic for a "proxy" function:
        // 1. Very small (few tokens)
        // 2. Exactly one outbound neighbor
        // 3. Code contains a return or call to that neighbor
        if (node.tokens < 50 && node.neighbors.size === 1) {
            const neighbor = Array.from(node.neighbors)[0];
            const escapedNeighbor = neighbor.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b${escapedNeighbor}\\b`);

            if (regex.test(node.code)) {
                // This is a proxy. Redirect all nodes that point to this proxy 
                // to point directly to the neighbor instead.
                for (const [otherName, otherNode] of nodes) {
                    if (otherNode.neighbors.has(name)) {
                        otherNode.neighbors.delete(name);
                        if (otherName !== neighbor) {
                            otherNode.neighbors.add(neighbor);
                        }
                        // SAFE METADATA UPDATE: Refresh all fields
                        otherNode.neighborCount = otherNode.neighbors.size;
                        otherNode.outbound = Array.from(otherNode.neighbors);
                    }
                }
                nodes.delete(name);
                collapsedCount++;
            }
        }
    }
    console.log(`    [+] Collapsed ${collapsedCount} proxy nodes.`);
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
    }
};


class CascadeGraph {
    constructor() {
        this.nodes = new Map(); // ChunkName -> { code, tokens, category, neighbors: Set }
        this.internalNameToChunkId = new Map(); // Name -> ChunkName
        this.totalTokens = 0;
        this.runtimeMap = {};
    }

    // Phase 0.5: Detect the names of common obfuscated helpers based on their shape
    detectRuntimeHelpers(ast) {
        const helpers = {};
        const statements = ast.program.body;

        console.log(`[*] Phase 0.5: Dynamically detecting runtime helpers...`);

        // We only need to check the top of the file where esbuild puts its runtime
        for (let i = 0; i < Math.min(statements.length, 500); i++) {
            const node = statements[i];
            if (node.type !== 'VariableDeclaration') continue;

            node.declarations.forEach(decl => {
                if (!decl.init) return;
                const code = generate(decl.init, { minified: true }).code;
                const name = decl.id.name;

                // 1. Detect Lazy Loader (w)
                // Fingerprint: (A,Q)=>()=>(A&&(Q=A(A=0)),Q)
                if (code.includes('A=0') && (code.includes('Q=A(') || code.includes('Q = A('))) {
                    helpers[name] = '__lazyInit';
                }

                // 2. Detect CommonJS Wrapper (U)
                // Fingerprint: (A,Q)=>()=>(Q||A((Q={exports:{}}).exports,Q),Q.exports)
                if (code.includes('exports:{}') && (code.includes('Q.exports') || code.includes('Q .exports'))) {
                    helpers[name] = '__commonJS';
                }

                // 3. Detect ESM Export (U5)
                // Fingerprint: (A,Q)=>{for(var B in Q)Object.defineProperty(A,B,{...})}
                if (code.includes('defineProperty') && code.includes('enumerable:true') && code.includes('configurable:true')) {
                    // Check if it looks like a loop assigning properties
                    if (code.includes('for(var') || code.includes('.map(')) {
                        helpers[name] = '__export';
                    }
                }

                // 4. Detect toESM Interop (c)
                if (code.includes('__esModule') && code.includes('getPrototypeOf') && code.includes('default:A')) {
                    helpers[name] = '__toESM';
                }
            });
        }

        console.log(`[*] Dynamically identified helpers:`, helpers);
        this.runtimeMap = helpers;
        return helpers;
    }

    // Phase 0.6: Detect the internal state object
    detectInternalState(ast) {
        let stateVarName = null;
        const statements = ast.program.body;

        console.log(`[*] Phase 0.6: Detecting internal state object via mutation patterns (Shallow Scan)...`);

        const mutationCounts = new Map();

        // Shallow scan only for performance and to avoid deep recursion crashes
        for (const node of statements) {
            if (node.type === 'ExpressionStatement' && node.expression.type === 'AssignmentExpression') {
                const expr = node.expression;
                if (expr.left.type === 'MemberExpression' && expr.left.object.type === 'Identifier') {
                    const objName = expr.left.object.name;
                    mutationCounts.set(objName, (mutationCounts.get(objName) || 0) + 1);
                }
            } else if (node.type === 'VariableDeclaration') {
                node.declarations.forEach(decl => {
                    if (decl.init && decl.init.type === 'CallExpression') {
                        decl.init.arguments.forEach(arg => {
                            if (arg.type === 'Identifier') {
                                mutationCounts.set(arg.name, (mutationCounts.get(arg.name) || 0) + 0.5);
                            }
                        });
                    }
                });
            }
        }

        let bestCandidate = null;
        let maxScore = 0;

        for (const [name, score] of mutationCounts.entries()) {
            if (score > maxScore && name.length <= 4) {
                maxScore = score;
                bestCandidate = name;
            }
        }

        if (bestCandidate && maxScore > 2) { // Lowered threshold for shallow scan
            stateVarName = bestCandidate;
            console.log(`[*] Detected potential internal state variable: ${stateVarName} (Shallow Score: ${maxScore})`);
            this.runtimeMap[stateVarName] = 'INTERNAL_STATE';
        }

        return stateVarName;
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
    identifyNodes(ast, code) {
        console.log(`[*] Phase 1: Identifying Chunks...`);
        let statements = ast.program.body;

        // --- IIFE UNWRAPPING (Drill-down) ---
        while (statements.length === 1) {
            const node = statements[0];
            if (node.type === 'ExpressionStatement') {
                let expr = node.expression;
                // Handle !function(){}() or (function(){})()
                if (expr.type === 'UnaryExpression' && expr.operator === '!') expr = expr.argument;

                if (expr.type === 'CallExpression') {
                    const callee = expr.callee;
                    if (callee.type === 'FunctionExpression' || callee.type === 'ArrowFunctionExpression') {
                        console.log(`[*] Phase 1.1: Unwrapping IIFE wrapper...`);
                        statements = callee.body.type === 'BlockStatement' ? callee.body.body : [callee.body];
                        continue;
                    }
                }
            }
            break;
        }

        let currentChunkNodes = [];
        let currentTokens = 0;
        let currentCategory = 'unknown';
        let chunkIndex = 1;
        const seenHashes = new Map(); // hash -> chunkName

        const finalizeChunk = () => {
            if (currentChunkNodes.length === 0) return;

            // Use slicing from original code for stability and speed
            const firstNode = currentChunkNodes[0];
            const lastNode = currentChunkNodes[currentChunkNodes.length - 1];
            let chunkCode = code.slice(firstNode.start, lastNode.end);

            // --- HEADER PRESERVATION ---
            // If this is the first chunk, grab any leading comments from the very top of the file
            if (chunkIndex === 1) {
                const prefix = code.slice(0, firstNode.start);
                // Keep only the comments and whitespace, discard skipped IIFE wrappers
                const headerLines = prefix.split('\n').filter(line => {
                    const trimmed = line.trim();
                    return trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.includes('*/') || trimmed === '';
                });
                if (headerLines.length > 0) {
                    chunkCode = headerLines.join('\n').trim() + '\n\n' + chunkCode;
                }
            }

            // --- DE-DUPLICATION ---
            const hash = crypto.createHash('md5').update(chunkCode).digest('hex');
            if (seenHashes.has(hash)) {
                const originalName = seenHashes.get(hash);
                // Mark this as an alias in the internal mapping, but don't create a new node
                currentChunkNodes.forEach(node => {
                    this.extractInternalNames(node).forEach(name => {
                        this.internalNameToChunkId.set(name, originalName);
                    });
                });
                currentChunkNodes = [];
                currentTokens = 0;
                return;
            }

            const chunkName = `chunk${chunkIndex.toString().padStart(3, '0')}`;
            seenHashes.set(hash, chunkName);

            const startLine = currentChunkNodes[0].loc?.start.line || 0;
            const endLine = currentChunkNodes[currentChunkNodes.length - 1].loc?.end.line || 0;

            let suggestedName = null;
            let category = currentCategory;
            let kbInfo = null;
            let errorSignature = null;
            const hints = [];
            const hasTengu = chunkCode.toLowerCase().includes('tengu_');

            const entrySignals = [
                'process.argv',
                '#!/usr/bin/env',
                'cliMain',
                'commander',
                '.parse(process.argv)'
            ];
            const entrySignalCount = entrySignals.filter(s => chunkCode.includes(s)).length;

            // --- NEW: ENHANCED KB METADATA ---
            if (KB && !IS_BOOTSTRAP) {
                // 1. Check File Anchors
                if (KB.file_anchors) {
                    for (const anchor of KB.file_anchors) {
                        // Weighted keywords logic
                        let weightedSum = 0;
                        const triggerKeywords = Array.isArray(anchor.trigger_keywords) ? anchor.trigger_keywords : [];

                        triggerKeywords.forEach(kwEntry => {
                            const kw = typeof kwEntry === 'string' ? kwEntry : kwEntry.word;
                            const weight = typeof kwEntry === 'string' ? 1 : (kwEntry.weight || 1);
                            if (chunkCode.includes(kw)) {
                                weightedSum += weight;
                            }
                        });

                        const threshold = anchor.threshold || Math.max(2, Math.floor(triggerKeywords.length * 0.6));

                        if (weightedSum >= threshold) {
                            suggestedName = anchor.suggested_name.replace(/`/g, '');
                            category = 'priority';
                            kbInfo = {
                                suggested_path: anchor.suggested_path,
                                description: anchor.description,
                                weighted_sum: weightedSum
                            };
                            console.log(`    [+] KB Match: Found ${suggestedName} (Score: ${weightedSum})`);
                            break;
                        }
                    }
                }

                // 2. Scan for Name Hints (Logic Anchors)
                if (KB.name_hints) {
                    for (const hint of KB.name_hints) {
                        if (hint.logic_anchor && hint.logic_anchor.length > 0 && chunkCode.includes(hint.logic_anchor)) {
                            hints.push({
                                suggested_name: hint.suggested_name.replace(/`/g, ''),
                                logic_anchor: hint.logic_anchor
                            });
                        }
                    }
                }

                // 3. Scan for Error Anchors
                if (KB.error_anchors) {
                    for (const anchor of KB.error_anchors) {
                        if (chunkCode.includes(anchor.content)) {
                            errorSignature = anchor.role;
                            console.log(`    [!] Error Anchor: Found ${anchor.role}`);
                        }
                    }
                }
            }

            // Detect Structural Signals (DNA)
            const hasGenerator = code.includes('async function*') || code.includes('yield*');
            const hasStateMutator = code.includes('INTERNAL_STATE.') && code.includes('=');
            // ---------------------------------

            // Intelligent Naming Logic (Fallback)
            if (!suggestedName) {
                const collectedStrings = [];
                currentChunkNodes.forEach(node => {
                    // Dead code removal is now done globally in Phase 0.9 with full scope awareness
                    findStrings(node, collectedStrings);
                });

                for (const val of collectedStrings) {
                    const isPath = val.includes('/') || val.includes('\\');
                    const isModule = val.includes('@') || val.endsWith('.js') || val.endsWith('.ts');
                    const isClaudeSignal = val.startsWith('tengu_') || val.startsWith('claude_') || val.toLowerCase().includes('anthropic');

                    if (isPath || isModule || isClaudeSignal) {
                        if (!suggestedName || val.length < suggestedName.length) {
                            if (val.length > 3 && val.length < 60) suggestedName = val;
                        }
                    }
                }
            }

            const tokens = encode(chunkCode).length;

            // Generate structural AST for ML anchoring
            const structuralAST = currentChunkNodes.map(simplifyAST);

            this.nodes.set(chunkName, {
                id: chunkIndex,
                name: chunkName,
                code: chunkCode,
                tokens,
                startLine,
                endLine,
                structuralAST, // Store for saveResults
                neighbors: new Set(),
                suggestedFilename: suggestedName,
                kb_info: kbInfo,
                hints: hints,
                error_signature: errorSignature,
                hasGenerator: currentChunkNodes.some(n => n.type === 'YieldExpression' || n.delegate),
                hasStateMutator: currentChunkNodes.some(n => n.type === 'AssignmentExpression' && n.left.object?.name === 'INTERNAL_STATE'),
                hasTengu: hasTengu,
                entrySignalCount: entrySignalCount,
                hasNetwork: chunkCode.toLowerCase().includes('http') || chunkCode.toLowerCase().includes('https') || chunkCode.toLowerCase().includes('socket'),
                hasFS: chunkCode.toLowerCase().includes('fs.') || chunkCode.toLowerCase().includes('path.') || chunkCode.toLowerCase().includes('filename'),
                hasCrypto: chunkCode.toLowerCase().includes('crypto') || chunkCode.toLowerCase().includes('hash'),
                isUI: chunkCode.toLowerCase().includes('react') || chunkCode.toLowerCase().includes('ink') || chunkCode.toLowerCase().includes('box'),
                startsWithImport: currentChunkNodes.some(node => {
                    const nodeCode = code.slice(node.start, node.end);
                    return (
                        node.type === 'ImportDeclaration' ||
                        nodeCode.includes('require(') ||
                        nodeCode.includes('_toESM(') ||
                        nodeCode.includes('__commonJS(') ||
                        nodeCode.includes('import.meta')
                    );
                })
            });

            if (suggestedName) {
                const sanitized = suggestedName.replace(/[^a-zA-Z0-9]/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '');
                if (sanitized.length > 1) {
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
        const wrapperNames = new Set(
            Object.keys(this.runtimeMap).filter(name =>
                this.runtimeMap[name] === '__commonJS' || this.runtimeMap[name] === '__lazyInit'
            )
        );

        if (wrapperNames.size > 0) console.log(`[*] Using detected module wrappers for splitting: ${Array.from(wrapperNames).join(', ')}`);

        statements.forEach((node) => {
            const nodeCodeApprox = code.slice(node.start, node.end).toLowerCase();
            // Performance Optimization: Rough token estimate (length / 2.5)
            // Obfuscated code has many short identifiers, making length/4 too optimistic.
            const nodeTokensApprox = nodeCodeApprox.length / 2.5;
            const hasSignal = SIGNAL_KEYWORDS.some(kw => nodeCodeApprox.includes(kw));
            const nodeCategory = hasSignal ? 'priority' : 'vendor';

            const isImport = node.type === 'ImportDeclaration' ||
                (node.type === 'VariableDeclaration' && nodeCodeApprox.includes('require('));

            const isModuleWrapperCall = node.type === 'VariableDeclaration' &&
                node.declarations.length === 1 &&
                node.declarations[0].init &&
                node.declarations[0].init.type === 'CallExpression' &&
                wrapperNames.has(node.declarations[0].init.callee.name);

            const isBanal = node.type === 'ExpressionStatement' &&
                node.expression.type === 'StringLiteral' &&
                (node.expression.value === 'use strict' || node.expression.value.startsWith('__esModule'));

            const isUtility = node.type === 'VariableDeclaration' &&
                node.declarations.some(d => d.id.name && (NATIVE_PROPS.includes(d.id.name) || d.id.name.startsWith('_')));

            const isInsideLargeDeclaration = node.type === 'VariableDeclaration' && nodeTokensApprox > 250;

            // Only allow splitting at the Top Level of the program
            const isTopLevel = statements.includes(node);

            const chunkThreshold = parseInt(process.env.CHUNKING_TOKEN_THRESHOLD) || 2000;
            const shouldSplit = currentChunkNodes.length > 0 && isTopLevel && !isInsideLargeDeclaration && (
                (isImport && currentChunkHasContent) ||
                (isModuleWrapperCall && currentTokens > 3000) ||
                (nodeCategory === 'priority' && currentCategory === 'vendor' && currentTokens > 1500) ||
                (currentTokens + nodeTokensApprox > chunkThreshold) || // threshold*4 = approx 8k tokens
                (isUtility && currentTokens > chunkThreshold)
            );

            if (shouldSplit) {
                finalizeChunk();
                currentChunkHasContent = false;
            }

            currentChunkNodes.push(node);
            currentTokens += nodeTokensApprox;

            if (!isImport && !isBanal) {
                currentChunkHasContent = true;
            }

            if (nodeCategory === 'priority') currentCategory = 'priority';
            else if (currentCategory === 'unknown') currentCategory = 'vendor';
        });

        finalizeChunk();
    }

    // Step B: Neighbor Detection (The Edges)
    detectNeighbors() {
        console.log(`[*] Phase 2: Neighbor Detection (AST-based Edges)...`);
        const internalNamesSet = new Set(this.internalNameToChunkId.keys());

        for (const [chunkName, nodeData] of this.nodes) {
            try {
                const ast = parser.parse(nodeData.code, {
                    sourceType: 'module',
                    plugins: ['jsx', 'typescript', 'decorators-legacy', 'classProperties', 'dynamicImport', 'topLevelAwait']
                });

                traverse(ast, {
                    Identifier(p) {
                        const name = p.node.name;
                        if (internalNamesSet.has(name) && p.isReferencedIdentifier()) {
                            const targetChunk = this.internalNameToChunkId.get(name);
                            if (chunkName !== targetChunk) {
                                nodeData.neighbors.add(targetChunk);
                            }
                        }
                    }
                });
            } catch (err) {
                // Fallback to regex if parse fails (unlikely since we generated this code)
                const internalNames = Array.from(internalNamesSet);
                internalNames.forEach(definedName => {
                    const targetChunk = this.internalNameToChunkId.get(definedName);
                    if (chunkName === targetChunk) return;

                    if (nodeData.code.includes(definedName)) {
                        const escapedName = definedName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                        const regex = new RegExp(`\\b${escapedName}\\b`);
                        if (regex.test(nodeData.code)) {
                            nodeData.neighbors.add(targetChunk);
                        }
                    }
                });
            }
        }
    }

    // Step C: Markov Chain Analysis
    applyMarkovAnalysis() {
        console.log(`[*] Phase 3: Markov Chain Centrality Calculation (Weighted PageRank variant)...`);
        const names = Array.from(this.nodes.keys());
        const size = names.length;
        if (size === 0) return;
        const nameToIndex = new Map(names.map((name, idx) => [name, idx]));

        const dampingFactor = parseFloat(process.env.MARKOV_DAMPING_FACTOR) || 0.85;
        const teleportProbability = (1 - dampingFactor) / size;
        const inDegree = new Map();

        // Pre-calculate edge weights
        // Edges pointing to "priority" (founder) chunks carry more weight
        const edgeWeights = new Map(); // chunkName -> {neighbors: Array<{name, weight}>, totalWeight: number}
        this.nodes.forEach((node, name) => {
            let totalWeight = 0;
            const weightedNeighbors = [];
            node.neighbors.forEach(neighborName => {
                const neighborNode = this.nodes.get(neighborName);
                const weight = (neighborNode && neighborNode.category === 'priority') ? 2.0 : 1.0;
                weightedNeighbors.push({ name: neighborName, weight });
                totalWeight += weight;
            });
            edgeWeights.set(name, { neighbors: weightedNeighbors, totalWeight });

            // Still track in-degree for the final score adjustment
            node.neighbors.forEach(neighbor => inDegree.set(neighbor, (inDegree.get(neighbor) || 0) + 1));
        });

        let scores = new Array(size).fill(1 / size);

        const founderIndices = names
            .map((name, idx) => (this.nodes.get(name).category === 'priority' ? idx : -1))
            .filter(idx => idx !== -1);
        const redistributionTargets = founderIndices.length > 0 ? founderIndices : Array.from({ length: size }, (_, i) => i);

        for (let iter = 0; iter < 100; iter++) { // Optimized from 200 iterations
            let nextScores = new Array(size).fill(teleportProbability);
            let maxDiff = 0;

            for (let i = 0; i < size; i++) {
                const name = names[i];
                const nodeData = edgeWeights.get(name);
                const { neighbors, totalWeight } = nodeData;

                if (totalWeight > 0) {
                    const availableProbability = scores[i] * dampingFactor;
                    neighbors.forEach(({ name: neighborName, weight }) => {
                        const neighborIdx = nameToIndex.get(neighborName);
                        if (neighborIdx !== undefined) {
                            nextScores[neighborIdx] += (availableProbability * weight) / totalWeight;
                        }
                    });
                } else {
                    // Hybrid Teleportation for Sink nodes: 85% to founders, 15% globally
                    const sinkContribution = (scores[i] * dampingFactor);
                    const founderPart = (sinkContribution * 0.85) / (redistributionTargets.length || 1);
                    const globalPart = (sinkContribution * 0.15) / (size || 1);

                    for (const targetIdx of redistributionTargets) {
                        nextScores[targetIdx] += founderPart;
                    }
                    for (let g = 0; g < size; g++) {
                        nextScores[g] += globalPart;
                    }
                }
            }

            // Convergence Check
            for (let i = 0; i < size; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(nextScores[i] - scores[i]));
            }

            scores = nextScores;
            if (maxDiff < 1e-10) { // Tighter convergence threshold
                console.log(`    [*] Markov Centrality converged after ${iter + 1} iterations (diff: ${maxDiff.toExponential(2)})`);
                break;
            }
        }

        names.forEach((name, i) => {
            const node = this.nodes.get(name);
            const inCount = inDegree.get(name) || 0;
            const outCount = node.neighbors.size;
            // Centrality + (Out-Degree / (In-Degree + 1)) adjustment
            node.score = scores[i] + (outCount / (inCount + 1)) * 0.01;
        });
    }

    // Phase 4: Re-classify based on aggregated chunk data (NN + Relational Identity)
    classify(versionPath) {
        console.log(`[*] Phase 4: Final Classification (NN + Relational Identity)...`);

        // Load the mapping we just generated in Phase 3.5
        const mappingPath = path.join(versionPath, 'metadata', 'mapping.json');
        const nnMapping = fs.existsSync(mappingPath) ? JSON.parse(fs.readFileSync(mappingPath, 'utf8')) : { processed_chunks: [], variables: {} };

        // Optimize variable lookup: map source -> entry
        const varSourceMap = new Map();
        if (nnMapping.variables) {
            for (const entry of Object.values(nnMapping.variables)) {
                if (entry.source) varSourceMap.set(entry.source, entry);
            }
        }

        // 1. Pre-calculate Metrics (In-Degree)
        const inDegree = new Map();
        for (const [name, node] of this.nodes) {
            for (const neighbor of node.neighbors) {
                inDegree.set(neighbor, (inDegree.get(neighbor) || 0) + 1);
            }
        }

        // 2. Identify "Founder Seeds" (Keywords + Async Generators + State DNA)
        const familySet = new Set();
        for (const [name, node] of this.nodes) {
            const code = node.code;
            const hasTengu = code.toLowerCase().includes('tengu_');
            const hasGenerator = node.hasGenerator;
            const isStateMutator = node.hasStateMutator;
            const hasErrorSignature = !!node.error_signature;

            if (hasTengu || hasGenerator || isStateMutator || hasErrorSignature) {
                familySet.add(name);
                node.category = 'founder';
            }
        }
        console.log(`    [+] Identified ${familySet.size} founder chunks.`);

        // 3. Spreading Activation (Tengu Spreading / Neighbor Collision)
        let changed = true;
        let iteration = 0;
        while (changed && iteration < 10) {
            iteration++;
            let startSize = familySet.size;
            for (const [name, node] of this.nodes) {
                if (familySet.has(name)) continue;

                const neighbors = Array.from(node.neighbors);
                const familyNeighbors = neighbors.filter(n => familySet.has(n));

                const spreadingRatio = parseFloat(process.env.SPREADING_THRESHOLD_RATIO) || 0.3;
                const spreadingCount = parseInt(process.env.SPREADING_THRESHOLD_COUNT) || 2;

                if (neighbors.length > 0 && (familyNeighbors.length / neighbors.length >= spreadingRatio || familyNeighbors.length >= spreadingCount)) {
                    familySet.add(name);
                }
            }
            changed = familySet.size > startSize;
        }
        console.log(`    [+] Spreading complete: ${familySet.size} chunks in family set after ${iteration} iterations.`);

        // 4. Role Assignment via Capability & Modularity
        for (const [name, node] of this.nodes) {
            const inCount = inDegree.get(name) || 0;
            const outCount = node.neighbors.size;
            const isFamily = familySet.has(name);
            const isAnchored = nnMapping.processed_chunks.includes(name);
            const matchMeta = nnMapping.matches ? nnMapping.matches[name] : null;
            const isLibraryMatch = matchMeta ? !!matchMeta.is_library_match : false;

            if (isAnchored && isLibraryMatch) {
                // Library match confirmed by anchor stage metadata
                node.category = 'vendor';
                node.label = 'LIBRARY_MATCH';

                // Try to find the suggested name from anchored variables (Optimized)
                const matchEntry = varSourceMap.get(name);
                if (matchEntry) node.role = `LIB: ${matchEntry.name}`;

            } else if (isAnchored && !isLibraryMatch) {
                if (!node.label) node.label = 'ANCHOR_MATCH';
                if (matchMeta?.label) node.role = `ANCHOR: ${matchMeta.label}`;
            } else if (isFamily && node.category !== 'founder') {
                node.category = 'family';
            } else if (!isFamily) {
                node.category = 'vendor';
            }

            const vendorIn = parseInt(process.env.VENDOR_LIBRARY_IN_DEGREE) || 15;
            const vendorOut = parseInt(process.env.VENDOR_LIBRARY_OUT_DEGREE) || 5;
            const coreIn = parseInt(process.env.CORE_ORCHESTRATOR_IN_DEGREE) || 5;
            const coreOut = parseInt(process.env.CORE_ORCHESTRATOR_OUT_DEGREE) || 5;

            if (!isFamily && inCount > vendorIn && outCount < vendorOut) {
                node.label = 'VENDOR_LIBRARY';
            } else if (isFamily && inCount > coreIn && outCount > coreOut) {
                node.label = 'CORE_ORCHESTRATOR';
            }

            let role = node.role || 'MODULE';

            if (node.hasGenerator) {
                role = 'STREAM_ORCHESTRATOR';
            } else if (node.code.includes('child_process') || node.code.includes('spawn')) {
                role = 'SHELL_EXECUTOR';
            } else if (node.code.includes('anthropic-beta') || node.code.includes('https://api.anthropic.com')) {
                role = 'API_CLIENT';
            } else if (node.code.includes('statsig')) {
                role = 'FEATURE_FLAGS';
            } else if (node.code.includes('Box') || node.code.includes('Text') || node.code.includes('ink')) {
                role = 'UI_COMPONENT';
            } else if (node.error_signature) {
                role = node.error_signature;
            } else if (isFamily) {
                role = 'CORE_LOGIC';
            }

            node.role = role;

            node.state_touchpoints = [];
            const stateMatches = node.code.match(/INTERNAL_STATE\.(\w+)/g);
            if (stateMatches) {
                node.state_touchpoints = [...new Set(stateMatches.map(m => m.split('.')[1]))];
            }

            if (node.code.includes('totalCostUSD')) node.state_touchpoints.push('FINANCIAL_LOGIC');
            if (node.code.includes('invokedSkills')) node.state_touchpoints.push('TOOL_DISPATCHER');
        }
    }

    saveResults(versionDir) {
        const chunksDir = path.join(versionDir, 'chunks');
        const metadataDir = path.join(versionDir, 'metadata');

        [chunksDir, metadataDir].forEach(d => {
            if (!fs.existsSync(d)) {
                fs.mkdirSync(d, { recursive: true });
            }
        });

        // SOFT MERGE: Preserve valuable metadata from previous runs
        let existingMetadata = [];
        if (fs.existsSync(path.join(metadataDir, 'graph_map.json'))) {
            try {
                const raw = JSON.parse(fs.readFileSync(path.join(metadataDir, 'graph_map.json')));
                existingMetadata = Array.isArray(raw) ? raw : (raw.chunks || []);
                console.log(`[*] Soft Merge: Loaded ${existingMetadata.length} existing chunks for metadata preservation.`);
            } catch (e) {
                console.warn(`[!] Soft Merge: Failed to read existing metadata. Starting fresh.`);
            }
        }

        // Create the flat array format expected by the pipeline
        const metadata = [];
        const fileRanges = []; // Keep fileRanges for graph_map.js, but not for graph_map.json
        for (const [name, node] of this.nodes) {
            let fileName = `${name}.js`;
            if (node.displayName) {
                fileName = `${name}_${node.displayName}.js`;
            }

            fs.writeFileSync(path.join(chunksDir, fileName), node.code);

            // Soft Match Logic
            const oldEntry = existingMetadata.find(e => e.name === name);

            const metaEntry = {
                name,
                displayName: node.displayName || (oldEntry ? oldEntry.displayName : null),
                file: `chunks/${fileName}`,
                tokens: node.tokens || 0,
                score: node.score || 0,
                centrality: node.score || 0, // Corrected name
                role: node.role !== 'MODULE' ? node.role : (oldEntry ? oldEntry.role : 'MODULE'),
                category: node.category || 'unknown',
                label: node.label || null,
                outbound: Array.from(node.neighbors),
                kb_info: node.kb_info,
                hints: node.hints,
                // CRITICAL: Preserve expensive LLM-generated paths
                proposedPath: node.proposedPath || (oldEntry ? oldEntry.proposedPath : null),
                suggestedFilename: node.suggestedFilename || (oldEntry ? oldEntry.suggestedFilename : null),
                isGoldenMatch: node.isGoldenMatch || (oldEntry ? oldEntry.isGoldenMatch : false),

                hasTengu: node.hasTengu || false,
                hasGenerator: node.hasGenerator || false,
                hasStateMutator: node.hasStateMutator || false,
                entrySignalCount: node.entrySignalCount || 0,
                error_signature: node.error_signature || null,
                startsWithImport: node.startsWithImport || false,
                startLine: node.startLine || 0,
                endLine: node.endLine || 0
            };
            metadata.push(metaEntry);

            // Group by suggested path or display name for the summary
            const suggestedPath = node.kb_info?.suggested_path || node.displayName || 'unknown';
            let rangeEntry = fileRanges.find(r => r.path === suggestedPath);
            if (!rangeEntry) {
                rangeEntry = { path: suggestedPath, ranges: [] };
                fileRanges.push(rangeEntry);
            }

            // Try to merge adjacent ranges
            const lastRange = rangeEntry.ranges[rangeEntry.ranges.length - 1];
            if (lastRange && node.startLine <= lastRange.end + 5) { // 5 line buffer for whitespace/comments
                lastRange.end = Math.max(lastRange.end, node.endLine);
            } else {
                rangeEntry.ranges.push({ start: node.startLine, end: node.endLine });
            }
        }

        const finalOutput = {
            chunks: metadata,
            file_ranges: fileRanges.sort((a, b) => a.ranges[0].start - b.ranges[0].start)
        };

        fs.writeFileSync(
            path.join(metadataDir, 'graph_map.json'),
            JSON.stringify(finalOutput, null, 2)
        );

        fs.writeFileSync(
            path.join(metadataDir, 'graph_map.js'),
            `window.GRAPH_DATA = ${JSON.stringify(finalOutput, null, 2)};`
        );

        // Save Structural ASTs for ML anchoring
        const structuralASTs = {};
        for (const [name, node] of this.nodes) {
            structuralASTs[name] = {
                ast: node.structuralAST,
                kb_info: node.kb_info,
                hints: node.hints,
                category: node.category,
                role: node.role
            };
            // Optimization: Explicitly clear large temporary structures from the node object 
            // to free up memory before the next bulky operation (Neural Anchoring).
            node.code = null;
            node.structuralAST = null;
        }
        fs.writeFileSync(
            path.join(metadataDir, 'simplified_asts.json'),
            JSON.stringify(structuralASTs)
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
    console.log(`[*] Phase 0: Preprocessing with Webcrack (De-proxifying)...`);
    try {
        const result = await webcrack(code, {
            unminify: false, // Let Babel handle unminification for consistency
            mangle: false
        });
        code = result.code;
        console.log(`[*] Preprocessing complete. Proxy functions resolved.`);
    } catch (err) {
        console.error(`[!] Webcrack failed: ${err.message}. Proceeding with raw code.`);
    }

    let ast = parser.parse(code, {
        sourceType: 'module',
        plugins: ['jsx', 'typescript', 'decorators-legacy', 'classProperties', 'dynamicImport', 'topLevelAwait'],
        errorRecovery: true
    });

    const graph = new CascadeGraph();

    // NEW: Dynamic Helper Detection
    const runtimeMap = graph.detectRuntimeHelpers(ast);
    graph.detectInternalState(ast);
    // --- OPTIMIZED GLOBAL RENAMING PASS (Phases 0.7 & 0.8) ---
    const namesToRename = Object.keys(runtimeMap);
    if (namesToRename.length > 0) {
        console.log(`[*] Phase 0.7+: Applying non-recursive global renaming...`);

        // Custom non-recursive walker to prevent stack overflows on deep esbuild trees
        const stack = [ast];
        while (stack.length > 0) {
            const node = stack.pop();
            if (!node || typeof node !== 'object' || node._visited) continue;

            if (node.type === 'Identifier') {
                if (runtimeMap[node.name]) {
                    node.name = runtimeMap[node.name];
                }
            }

            // Push children to stack
            for (const key in node) {
                if (key === 'tokens' || key === 'comments' || key === 'loc' || key === 'start' || key === 'end') continue;
                const child = node[key];
                if (child && typeof child === 'object') {
                    if (Array.isArray(child)) {
                        for (let i = child.length - 1; i >= 0; i--) {
                            if (child[i] && typeof child[i] === 'object') stack.push(child[i]);
                        }
                    } else {
                        stack.push(child);
                    }
                }
            }
        }
    }

    graph.identifyNodes(ast, code);
    graph.detectNeighbors();
    // collapseProxyNodes is already metadata-based
    collapseProxyNodes(graph.nodes);
    graph.applyMarkovAnalysis();
    OUTPUT_BASE = path.join(OUTPUT_ROOT, version);
    console.log(`[*] Output directory: ${OUTPUT_BASE}`);

    graph.saveResults(OUTPUT_BASE);

    console.log(`\n[COMPLETE] Static Analysis Phase Done.`);
    console.log(`Phase 1-3 results saved to: ${OUTPUT_BASE}`);
    return { version, path: OUTPUT_BASE };
}

run().catch(err => {
    console.error(`\n[!] FATAL ERROR during analysis:`);
    console.error(err);
    process.exit(1);
});
