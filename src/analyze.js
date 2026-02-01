require('dotenv').config();
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const readline = require('readline');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const { encode } = require('gpt-tokenizer');
const { webcrack } = require('webcrack');
const { AAdecode, jjdecode } = require('./deobfuscation_helpers');
const t = require('@babel/types');
const https = require('https');
const crypto = require('crypto');

// Utility for inspecting AST nodes programmatically
const inspectNode = (node, depth = 0, maxDepth = 5, filePath = path.join(process.env.GEMINI_TEMP_DIR, 'ast_inspection_log.jsonl')) => {
    if (!node || depth > maxDepth) return;

    const relevantProps = {};
    if (node.type) relevantProps.type = node.type;
    if (node.name) relevantProps.name = node.name;
    if (node.value !== undefined) relevantProps.value = node.value;
    if (node.operator) relevantProps.operator = node.operator;
    if (node.kind) relevantProps.kind = node.kind;
    if (node.callee && node.callee.type) relevantProps.calleeType = node.callee.type;
    if (node.arguments && Array.isArray(node.arguments)) relevantProps.argumentsLength = node.arguments.length;
    if (node.declarations && Array.isArray(node.declarations)) relevantProps.declarationsLength = node.declarations.length;
    if (node.expression && node.expression.type) relevantProps.expressionType = node.expression.type;
    if (node.body && node.body.type) relevantProps.bodyType = node.body.type;
    if (node.body && node.body.body && Array.isArray(node.body.body)) relevantProps.bodyStatements = node.body.body.length;


    fs.appendFileSync(filePath, JSON.stringify({ depth, ...relevantProps }) + '\n');

    for (const key in node) {
        if (key.startsWith('_') || key === 'loc' || key === 'start' || key === 'end' || key === 'comments' || key === 'tokens' || key === 'extra' || key === 'parentNode') {
            continue;
        }
        const prop = node[key];
        if (Array.isArray(prop)) {
            prop.forEach(child => inspectNode(child, depth + 1, maxDepth, filePath));
        } else if (prop && typeof prop === 'object' && prop.type) {
            inspectNode(prop, depth + 1, maxDepth, filePath);
        }
    }
};

// --- 0. KNOWLEDGE BASE ---
const { loadKnowledgeBase } = require('./knowledge_base');
let KB = null;
const { kb: loadedKb, path: kbPath } = loadKnowledgeBase();
if (loadedKb) {
    KB = loadedKb;
    const kbLabel = kbPath ? path.basename(kbPath) : 'knowledge_base.json';
    console.log(`[*] Loaded Knowledge Base (${kbLabel}) with ${KB.file_anchors?.length || 0} file anchors.`);
}

// --- 1. CONFIGURATION ---
const PACKAGE_NAME = '@anthropic-ai/claude-code';
const ANALYSIS_DIR = './claude-analysis';
const OUTPUT_ROOT = './cascade_graph_analysis';
let OUTPUT_BASE = OUTPUT_ROOT; // Will be updated with version
const IS_BOOTSTRAP = process.argv.includes('--is-bootstrap');
const GCS_BUCKET = 'https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases';
const SIGNAL_KEYWORDS = ['anthropic', 'claude', 'mcp', 'agent', 'terminal', 'prompt', 'session', 'protocol', 'codeloop'];
const NATIVE_PROPS = ['toString', 'hasOwnProperty', 'constructor', 'prototype', 'call', 'apply', 'bind'];
const GLOBAL_VARS = ['console', 'window', 'document', 'process', 'module', 'require', 'exports', 'global', 'Buffer', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval'];
const GENERIC_KB_KEYWORDS = new Set([
    'react', 'react-dom', 'jsx', 'tsx', 'js',
    'column', 'row', 'flex', 'flex-start', 'flex-end', 'center',
    'box', 'text', 'view', 'data', 'div', 'span', 'classname', 'style'
]);
const LOW_SIGNAL_KB_KEYWORDS = new Set([
    'react', 'react-dom', 'lodash', 'lodash-es', 'zod', 'axios', 'ink',
    'chalk', 'commander', 'sentry', 'statsig', 'tree-sitter', 'jsdom'
]);
const KB_LOW_WEIGHT_MULTIPLIER = parseFloat(process.env.KB_LOW_WEIGHT_MULTIPLIER) || 0.1;

const JS_EXTENSIONS = new Set(['.js', '.mjs']);
const getFileExtension = filePath => path.extname(filePath || '').toLowerCase();
const isJsFile = filePath => JS_EXTENSIONS.has(getFileExtension(filePath));
const getPythonBin = () => {
    if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
    return process.platform === 'win32' ? 'python' : 'python3';
};

const getBinaryCliPath = version => path.join(ANALYSIS_DIR, version, 'binary', 'cli.js');
const getBinaryPath = version => path.join(ANALYSIS_DIR, version, 'binary', process.platform === 'win32' ? 'claude.exe' : 'claude');

const parseSemver = version => {
    const match = typeof version === 'string'
        ? version.trim().match(/^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$/)
        : null;
    if (!match) return null;
    return {
        major: Number(match[1]),
        minor: Number(match[2]),
        patch: Number(match[3]),
        prerelease: match[4] || ''
    };
};

const BUN_CUTOFF_VERSION = parseSemver('2.1.23');

const compareSemver = (a, b) => {
    if (!a || !b) return 0;
    if (a.major !== b.major) return a.major > b.major ? 1 : -1;
    if (a.minor !== b.minor) return a.minor > b.minor ? 1 : -1;
    if (a.patch !== b.patch) return a.patch > b.patch ? 1 : -1;
    if (a.prerelease && !b.prerelease) return -1;
    if (!a.prerelease && b.prerelease) return 1;
    return 0;
};

const fetchBuffer = (url, redirects = 0) => new Promise((resolve, reject) => {
    const request = https.get(url, res => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            if (redirects > 5) {
                reject(new Error(`Too many redirects for ${url}`));
                return;
            }
            resolve(fetchBuffer(res.headers.location, redirects + 1));
            return;
        }
        if (res.statusCode !== 200) {
            reject(new Error(`HTTP ${res.statusCode} for ${url}`));
            res.resume();
            return;
        }
        const chunks = [];
        res.on('data', chunk => chunks.push(chunk));
        res.on('end', () => resolve(Buffer.concat(chunks)));
    });
    request.on('error', reject);
});

const fetchText = async url => (await fetchBuffer(url)).toString('utf8').trim();

const downloadToFile = async (url, outputPath) => {
    const buffer = await fetchBuffer(url);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, buffer);
};

const isMusl = () => {
    if (process.platform !== 'linux') return false;
    try {
        const report = process.report && process.report.getReport ? process.report.getReport() : null;
        if (report && report.header && report.header.glibcVersionRuntime) return false;
    } catch (err) {
        // Ignore and fall back to file checks.
    }
    const muslFiles = [
        '/lib/libc.musl-x86_64.so.1',
        '/lib/libc.musl-aarch64.so.1'
    ];
    if (muslFiles.some(p => fs.existsSync(p))) return true;
    try {
        const lddPath = '/usr/bin/ldd';
        if (fs.existsSync(lddPath)) {
            return fs.readFileSync(lddPath, 'utf8').includes('musl');
        }
    } catch (err) {
        // Ignore.
    }
    return false;
};

const detectPlatformKey = () => {
    if (process.platform === 'win32') return 'win32-x64';

    let osKey;
    if (process.platform === 'darwin') osKey = 'darwin';
    else if (process.platform === 'linux') osKey = 'linux';
    else throw new Error(`Unsupported platform: ${process.platform}`);

    let archKey;
    if (process.arch === 'x64') archKey = 'x64';
    else if (process.arch === 'arm64') archKey = 'arm64';
    else throw new Error(`Unsupported architecture: ${process.arch}`);

    if (osKey === 'linux' && isMusl()) return `${osKey}-${archKey}-musl`;
    return `${osKey}-${archKey}`;
};

const getNpmLatestVersion = () => {
    try {
        return execSync(`npm view ${PACKAGE_NAME} version`).toString().trim();
    } catch (err) {
        console.warn(`[!] Failed to fetch NPM version: ${err.message}`);
        return null;
    }
};

const downloadClaudeBinary = async requestedVersion => {
    const rawVersion = requestedVersion || 'latest';
    const requestedSemver = parseSemver(rawVersion);
    const resolvedVersion = requestedSemver ? rawVersion : await fetchText(`${GCS_BUCKET}/latest`);
    const manifestUrl = `${GCS_BUCKET}/${resolvedVersion}/manifest.json`;
    const manifest = JSON.parse(await fetchText(manifestUrl));
    const platformKey = detectPlatformKey();
    const checksum = manifest?.platforms?.[platformKey]?.checksum;
    if (!checksum) {
        throw new Error(`Platform ${platformKey} not found in manifest for ${version}`);
    }

    const binaryName = process.platform === 'win32' ? 'claude.exe' : 'claude';
    const binaryUrl = `${GCS_BUCKET}/${resolvedVersion}/${platformKey}/${binaryName}`;
    const binaryPath = path.join(ANALYSIS_DIR, resolvedVersion, 'binary', binaryName);
    let needsDownload = true;

    if (fs.existsSync(binaryPath)) {
        const actual = crypto.createHash('sha256').update(fs.readFileSync(binaryPath)).digest('hex');
        if (actual === checksum) {
            needsDownload = false;
        } else {
            fs.unlinkSync(binaryPath);
        }
    }

    if (needsDownload) {
        await downloadToFile(binaryUrl, binaryPath);
        const actual = crypto.createHash('sha256').update(fs.readFileSync(binaryPath)).digest('hex');
        if (actual !== checksum) {
            fs.unlinkSync(binaryPath);
            throw new Error(`Checksum verification failed for ${binaryName}`);
        }
        if (process.platform !== 'win32') {
            fs.chmodSync(binaryPath, 0o755);
        }
    }

    return { binaryPath, version: resolvedVersion };
};

const extractBunEntrypoint = (version, binaryPath) => {
    const cliPath = getBinaryCliPath(version);
    if (fs.existsSync(cliPath)) {
        console.log(`[*] Bun entrypoint already extracted: ${cliPath}`);
        return cliPath;
    }

    const pythonBin = getPythonBin();
    try {
        execSync(`${pythonBin} scripts/extract_bun_bundle.py --version ${version}`, { stdio: 'inherit' });
        const binaryDir = path.dirname(cliPath);
        const searchDir = path.join(binaryDir, 'src', 'entrypoints');
        if (fs.existsSync(searchDir)) {
            const jscFile = fs.readdirSync(searchDir).find(f => f.endsWith('.jsc'));
            if (jscFile) {
                fs.renameSync(path.join(searchDir, jscFile), cliPath);
            }
        }
    } catch (err) {
        throw new Error(`Bun extraction failed: ${err.message}`);
    }

    if (!fs.existsSync(cliPath)) {
        throw new Error(`Expected Bun entrypoint not found after extraction: ${cliPath}`);
    }
    return cliPath;
};

async function ensureTargetExists() {
    let targetFile = null;
    let version = 'unknown';
    let targetType = 'js';
    const versionIdx = process.argv.indexOf('--version');
    const versionArg = versionIdx !== -1 ? process.argv[versionIdx + 1] : null;
    const rawArgs = process.argv.slice(2);
    const nonFlagArgs = [];
    for (let i = 0; i < rawArgs.length; i++) {
        const arg = rawArgs[i];
        if (arg === '--version') {
            i += 1;
            continue;
        }
        if (arg.startsWith('--')) continue;
        nonFlagArgs.push(arg);
    }
    const targetArg = nonFlagArgs[0];

    // 1. Check command line argument
    if (targetArg) {
        targetFile = path.resolve(targetArg);
        // Handle --version flag
        if (versionArg) {
            version = versionArg;
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

        targetType = isJsFile(targetFile) ? 'js' : 'binary';
        return { targetFile, version, targetType };
    }

    // 2. Use explicit --version if provided, otherwise fetch latest from NPM
    const npmLatestVersion = getNpmLatestVersion();
    if (versionArg) {
        version = versionArg;
    } else {
        if (npmLatestVersion) {
            console.log(`[*] Using latest NPM version of ${PACKAGE_NAME}: ${npmLatestVersion}`);
            version = npmLatestVersion;
        } else {
            // Fallback to locally existing if any
            if (fs.existsSync(ANALYSIS_DIR)) {
                const versions = fs.readdirSync(ANALYSIS_DIR).filter(v => /^\d+\.\d+\.\d+/.test(v)).sort().reverse();
                for (const v of versions) {
                    const potentialPath = path.join(ANALYSIS_DIR, v, 'cli.js');
                    if (fs.existsSync(potentialPath)) {
                        console.log(`[*] Using locally found version: ${v} (NPM check failed)`);
                        return { targetFile: potentialPath, version: v, targetType: 'js' };
                    }
                }
            }
        }
    }

    const desiredSemver = parseSemver(version);
    const isAfterBunCutoff = desiredSemver && compareSemver(desiredSemver, BUN_CUTOFF_VERSION) === 1;
    const shouldUseBinary = Boolean(
        (versionArg && !desiredSemver) ||
        isAfterBunCutoff
    );

    if (shouldUseBinary) {
        if (desiredSemver && isAfterBunCutoff) {
            const existingCli = getBinaryCliPath(version);
            if (fs.existsSync(existingCli)) {
                console.log(`[*] Found extracted Bun entrypoint for ${version}. Skipping download.`);
                return { targetFile: existingCli, version, targetType: 'js' };
            }
        }

        console.log(`[*] Using native binary download for ${version} (bun cutoff: 2.1.23)...`);
        try {
            const { binaryPath, version: resolvedVersion } = await downloadClaudeBinary(version);
            if (desiredSemver && compareSemver(parseSemver(resolvedVersion), BUN_CUTOFF_VERSION) === 1) {
                const cliPath = extractBunEntrypoint(resolvedVersion, binaryPath);
                return { targetFile: cliPath, version: resolvedVersion, targetType: 'js' };
            }
            return { targetFile: binaryPath, version: resolvedVersion, targetType: 'binary' };
        } catch (err) {
            console.error(`[!] Failed to download native binary: ${err.message}`);
            return { targetFile: null, version: 'unknown', targetType: 'binary' };
        }
    }

    // 3. Check if versioned folder already exists
    if (version !== 'unknown') {
        const versionPath = path.join(ANALYSIS_DIR, version);
        const candidates = [
            path.join(versionPath, 'cli.js'),
            path.join(versionPath, 'cli.mjs')
        ];
        const isBunVersion = desiredSemver && compareSemver(desiredSemver, BUN_CUTOFF_VERSION) === 1;
        if (isBunVersion) {
            const binaryCli = getBinaryCliPath(version);
            candidates.unshift(binaryCli);
            const binaryPath = getBinaryPath(version);
            if (!fs.existsSync(binaryCli) && fs.existsSync(binaryPath)) {
                try {
                    extractBunEntrypoint(version, binaryPath);
                } catch (err) {
                    throw new Error(`Failed to extract Bun entrypoint for ${version}: ${err.message}`);
                }
            }
        }
        const potentialPath = candidates.find(p => fs.existsSync(p));
        if (potentialPath) {
            console.log(`[*] Version ${version} already present. Skipping download.`);
            return { targetFile: potentialPath, version, targetType: 'js' };
        }
    }

    // 4. If not found or if version check succeeded but folder doesn't exist, download latest from npm
    if (version !== 'unknown') {
        process.stdout.write(`[*] Downloading version ${version} of ${PACKAGE_NAME}...`);
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

            const candidates = [
                path.join(versionPath, 'cli.js'),
                path.join(versionPath, 'cli.mjs')
            ];
            const downloadedCli = candidates.find(p => fs.existsSync(p));
            if (downloadedCli) {
                return { targetFile: downloadedCli, version, targetType: 'js' };
            }
        } catch (err) {
            console.error(`\n[!] Failed to download bundle: ${err.message}`);
            if (versionArg) {
                console.warn(`[!] Falling back to native binary download for ${version}...`);
                try {
                    const { binaryPath, version: resolvedVersion } = await downloadClaudeBinary(version);
                    return { targetFile: binaryPath, version: resolvedVersion, targetType: 'binary' };
                } catch (binaryErr) {
                    console.error(`[!] Native binary download failed: ${binaryErr.message}`);
                }
            }
        }
    }

    return { targetFile: null, version: 'unknown', targetType: 'js' };
}

/**
 * Robustly find all strings in a node tree without requiring a full Babel path/scope.
 */
function findStrings(node, strings = []) {
    if (!node) return strings;
    const stack = [node];
    while (stack.length > 0) {
        const current = stack.pop();
        if (!current) continue;
        if (Array.isArray(current)) {
            for (let i = current.length - 1; i >= 0; i--) {
                const child = current[i];
                if (child && typeof child === 'object') stack.push(child);
            }
            continue;
        }
        if (current.type === 'StringLiteral') {
            strings.push(current.value);
        }
        for (const key in current) {
            const child = current[key];
            if (child && typeof child === 'object') {
                if (Array.isArray(child)) {
                    for (let i = child.length - 1; i >= 0; i--) {
                        const item = child[i];
                        if (item && typeof item === 'object') stack.push(item);
                    }
                } else if (child.type) {
                    stack.push(child);
                }
            }
            if (key === 'callee' && child && child.type) {
                // Special handling for callee to avoid too much recursion
                // If it's a MemberExpression or SequenceExpression, we only care about the innermost callee
                if (child.type === 'MemberExpression' || child.type === 'SequenceExpression') {
                    // This is already handled by getInnermostCallee
                } else {
                    stack.push(child);
                }
            }
        }
    }
    return strings;
}

// --- NEW: STRUCTURAL AST EXPORT FOR ML ---

/**
 * Strips identifiers and values from a node to create a "shape-only" representation.
 * Now includes a literal hashing channel to preserve semantic signal.
 */
function simplifyAST(node) {
    if (!node || typeof node !== 'object') return null;

    const classifyLiteral = value => {
        if (typeof value !== 'string') return null;
        const trimmed = value.trim();
        if (!trimmed) return null;
        if (/^[A-Z0-9_]{3,}$/.test(trimmed)) return 'const';
        if (/(^\.{1,2}\/)|[\\]/.test(trimmed)) return 'path';
        if (/^[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]{1,6}$/.test(trimmed)) return 'path';
        return null;
    };

    const skipKeys = new Set(['loc', 'start', 'end', 'comments']);
    const resultMap = new Map();
    const stack = [{ node, visited: false }];

    while (stack.length > 0) {
        const frame = stack.pop();
        const current = frame.node;
        if (!current || typeof current !== 'object') continue;

        if (!frame.visited) {
            stack.push({ node: current, visited: true });
            if (Array.isArray(current)) {
                for (let i = current.length - 1; i >= 0; i--) {
                    const child = current[i];
                    if (child && typeof child === 'object') stack.push({ node: child, visited: false });
                }
            } else {
                for (const key in current) {
                    if (skipKeys.has(key)) continue;
                    const val = current[key];
                    if (val && typeof val === 'object') {
                        if (Array.isArray(val)) {
                            for (let i = val.length - 1; i >= 0; i--) {
                                const item = val[i];
                                if (item && typeof item === 'object') stack.push({ node: item, visited: false });
                            }
                        } else {
                            stack.push({ node: val, visited: false });
                        }
                    }
                }
            }
            continue;
        }

        if (Array.isArray(current)) {
            const arrResult = [];
            for (let idx = 0; idx < current.length; idx++) {
                const child = current[idx];
                const childResult = child && typeof child === 'object' ? resultMap.get(child) : null;
                if (childResult) {
                    childResult.slot = idx;
                    arrResult.push(childResult);
                }
            }
            resultMap.set(current, arrResult);
            continue;
        }

        const structure = { type: current.type };
        if (current.callee && current.callee.name) structure.call = current.callee.name;
        if (current.type === 'Identifier') structure.name = current.name;

        if (current.type === 'StringLiteral' || current.type === 'NumericLiteral' || current.type === 'BooleanLiteral') {
            const val = String(current.value);
            structure.valHash = crypto.createHash('md5').update(val).digest('hex').slice(0, 8);
            if (current.type === 'StringLiteral') {
                const kind = classifyLiteral(val);
                if (kind) structure.valKind = kind;
            }
        }

        const children = [];
        for (const key in current) {
            if (skipKeys.has(key)) continue;
            const val = current[key];
            if (val && typeof val === 'object') {
                const childResult = resultMap.get(val);
                if (!childResult) continue;
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

        if (children.length > 0) structure.children = children;
        resultMap.set(current, structure);
    }

    return resultMap.get(node) || null;
}

function collapseProxyNodes(nodes) {
    console.log(`[*] Phase 2.2: Collapsing Proxy Nodes (Metadata-Safe)...`);
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
            const escapedNeighbor = neighbor.replace(/[.*+?^${}()|[\\]/g, '\\$&');
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
    "IfStatement|ConditionalExpression": (path) => {
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
        this.structuralAstTempPath = path.join(
            os.tmpdir(),
            `cascade_simplified_asts_${Date.now()}_${Math.random().toString(16).slice(2)}.jsonl`
        );
        fs.writeFileSync(this.structuralAstTempPath, '');
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
        const chunkThreshold = parseFloat(process.env.CHUNK_THRESHOLD) || 1500;
        console.log(`[*] Phase 1: Static Analysis & Chunking (Chunk Threshold: ${chunkThreshold})...`);
        let statements = ast.program.body;

        // --- IIFE UNWRAPPING (Drill-down) ---
        // Helper to get the innermost callee in case of (0, func)() or (void 0, func)()
        const getInnermostCallee = (callee) => {
            if (callee.type === 'SequenceExpression') {
                return getInnermostCallee(callee.expressions[callee.expressions.length - 1]);
            } else if (callee.type === 'MemberExpression') {
                return getInnermostCallee(callee.property);
            }
            return callee;
        };

        while (statements.length === 1) {
            const node = statements[0];
            if (node.type === 'ExpressionStatement') {
                let expr = node.expression;
                // Handle !function(){}() or (function(){})()
                if (expr.type === 'UnaryExpression' && expr.operator === '!') expr = expr.argument;

                if (expr.type === 'CallExpression') {
                    const callee = getInnermostCallee(expr.callee); // Use the helper
                    if (callee.type === 'FunctionExpression' || callee.type === 'ArrowFunctionExpression') {
                        console.log(`[*] Phase 1.1: Unwrapping IIFE wrapper...`);
                        statements = callee.body.type === 'BlockStatement' ? callee.body.body : [callee.body];
                        continue;
                    }
                } else if (expr.type === 'FunctionExpression' || expr.type === 'ArrowFunctionExpression') {
                    console.log(`[*] Phase 1.1: Unwrapping top-level FunctionExpression...`);
                    statements = expr.body.type === 'BlockStatement' ? expr.body.body : [expr.body];
                    continue;
                }
            }
            break;
        }

        let currentChunkNodes = [];
        let currentTokens = 0;
        let currentCategory = 'unknown';
        let currentModuleId = null; // New: Track active module envelope
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
                // 1. Check Vendor Anchors (negative signal)
                if (KB.vendor_anchors || KB.known_packages) {
                    const vendorAnchors = Array.isArray(KB.vendor_anchors) ? KB.vendor_anchors : [];
                    const knownPackages = Array.isArray(KB.known_packages) ? KB.known_packages : [];
                    const combinedAnchors = vendorAnchors.concat(knownPackages.map(name => ({
                        logic_anchor: name,
                        suggested_name: name,
                        description: 'known_package'
                    })));

                    for (const anchor of combinedAnchors) {
                        const logicAnchor = typeof anchor === 'string'
                            ? anchor
                            : (anchor.logic_anchor || anchor.content || anchor.anchor);
                        const triggerKeywords = Array.isArray(anchor.trigger_keywords) ? anchor.trigger_keywords : [];

                        let matched = false;
                        if (logicAnchor && chunkCode.includes(logicAnchor)) matched = true;

                        if (!matched && triggerKeywords.length > 0) {
                            let weightedSum = 0;
                            let matchedSpecific = false;
                            triggerKeywords.forEach(kwEntry => {
                                const kw = typeof kwEntry === 'string' ? kwEntry : kwEntry.word;
                                const weight = typeof kwEntry === 'string' ? 1 : (kwEntry.weight || 1);
                                const kwNorm = String(kw).toLowerCase();
                                if (GENERIC_KB_KEYWORDS.has(kwNorm)) return;
                                const weightMultiplier = LOW_SIGNAL_KB_KEYWORDS.has(kwNorm) ? KB_LOW_WEIGHT_MULTIPLIER : 1;
                                if (chunkCode.includes(kw)) {
                                    weightedSum += weight * weightMultiplier;
                                    matchedSpecific = true;
                                }
                            });
                            const threshold = anchor.threshold || Math.max(2, Math.floor(triggerKeywords.length * 0.3));
                            if (matchedSpecific && weightedSum >= threshold) matched = true;
                        }

                        if (matched) {
                            if (!suggestedName && anchor.suggested_name) {
                                suggestedName = anchor.suggested_name.replace(/`/g, '');
                            }
                            category = 'vendor';
                            kbInfo = {
                                suggested_path: anchor.suggested_path,
                                description: anchor.description,
                                anchor_type: 'vendor',
                                anchor_match: logicAnchor || null
                            };
                            console.log(`    [+] KB Vendor Match: ${logicAnchor || suggestedName || 'vendor'}`);
                            break;
                        }
                    }
                }

                // 2. Check File Anchors (positive signal, but soft)
                if (KB.file_anchors && category !== 'vendor') {
                    for (const anchor of KB.file_anchors) {
                        // Weighted keywords logic
                        let weightedSum = 0;
                        const triggerKeywords = Array.isArray(anchor.trigger_keywords) ? anchor.trigger_keywords : [];

                        let matchedSpecific = false;
                        triggerKeywords.forEach(kwEntry => {
                            const kw = typeof kwEntry === 'string' ? kwEntry : kwEntry.word;
                            const weight = typeof kwEntry === 'string' ? 1 : (kwEntry.weight || 1);
                            const kwNorm = String(kw).toLowerCase();
                            if (GENERIC_KB_KEYWORDS.has(kwNorm)) return;
                            const weightMultiplier = LOW_SIGNAL_KB_KEYWORDS.has(kwNorm) ? KB_LOW_WEIGHT_MULTIPLIER : 1;
                            if (chunkCode.includes(kw)) {
                                weightedSum += weight * weightMultiplier;
                                matchedSpecific = true;
                            }
                        });

                        const threshold = anchor.threshold || Math.max(2, Math.floor(triggerKeywords.length * 0.3));

                        if (weightedSum >= threshold && matchedSpecific) {
                            suggestedName = anchor.suggested_name.replace(/`/g, '');
                            category = 'kb_priority';
                            kbInfo = {
                                suggested_path: anchor.suggested_path,
                                description: anchor.description,
                                weighted_sum: weightedSum,
                                anchor_type: 'file'
                            };
                            console.log(`    [+] KB Match: Found ${suggestedName} (Score: ${weightedSum})`);
                            break;
                        }
                    }
                }

                // 3. Scan for Name Hints (Logic Anchors)
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

                // 4. Scan for Error Anchors
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
                    const isClaudeSignal = val.startsWith('tengu_') || val.startsWith('claude_');

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
            const astEntry = {
                name: chunkName,
                ast: structuralAST,
                kb_info: kbInfo,
                hints: hints,
                category: category,
                role: null,
                moduleId: currentModuleId
            };
            fs.appendFileSync(this.structuralAstTempPath, `${JSON.stringify(astEntry)}\n`);

            this.nodes.set(chunkName, {
                id: chunkIndex,
                name: chunkName,
                code: chunkCode,
                tokens,
                startLine,
                endLine,
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
                moduleId: currentModuleId, // Track the hard module envelope ID
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

        const processStatements = (statementList, isTopLevelContext) => {
            statementList.forEach((node) => {
                const nodeCodeApprox = code.slice(node.start, node.end).toLowerCase();
                const nodeTokensApprox = nodeCodeApprox.length / 2.5;

                const isPotentialModule = node.type === 'VariableDeclaration' &&
                    node.declarations.length === 1 &&
                    node.declarations[0].init &&
                    node.declarations[0].init.type === 'CallExpression';

                // Check if the callee of the CallExpression is an identifier (e.g., `_create_common_module`)
                // or a MemberExpression/SequenceExpression that resolves to one.
                let actualCallee = isPotentialModule ? getInnermostCallee(node.declarations[0].init.callee) : null;
                // NEW LOGIC: Trigger recursive dive if it's a structural module wrapper
                const isStructuralModuleWrapper = isPotentialModule &&
                    actualCallee && actualCallee.type === 'Identifier' &&
                    node.declarations[0].init.arguments.some(arg => arg.type === 'ArrowFunctionExpression' || arg.type === 'FunctionExpression');
                
                // BUN FIX: If we find a giant module wrapper, recurse into it instead of treating it as one node.
                // This is now based on structure, not name, and is triggered by a CallExpression with a function argument.
                if (isStructuralModuleWrapper && nodeTokensApprox > chunkThreshold && isTopLevelContext) {
                    const callee = node.declarations[0].init;
                    const moduleFunc = callee.arguments.find(arg => arg.type === 'ArrowFunctionExpression' || arg.type === 'FunctionExpression');

                    if (moduleFunc && moduleFunc.body.type === 'BlockStatement') {
                        console.log(`    [+] Detected large structural module wrapper. Diving in...`);
                        finalizeChunk();
                        currentChunkHasContent = false;

                        const moduleStatements = moduleFunc.body.body;
                        // Create a chunk for the wrapper itself, but without the body.
                        const declCopy = { ...node.declarations[0] };
                        const funcCopy = { ...moduleFunc, body: t.blockStatement([]) };
                        const argsCopy = callee.arguments.map(arg => arg === moduleFunc ? funcCopy : arg);
                        const calleeCopy = { ...callee, arguments: argsCopy };
                        declCopy.init = calleeCopy;
                        const wrapperNode = { ...node, declarations: [declCopy] };

                        currentChunkNodes.push(wrapperNode);
                        currentTokens += 50; // Approx token count for wrapper
                        finalizeChunk();

                        // Recurse into the actual module body
                        processStatements(moduleStatements, false);
                        finalizeChunk();
                        currentChunkHasContent = false;
                        return; // Done with this node, continue to the next in forEach.
                    }
                }

                // --- MODULE ENVELOPE DETECTION (Hard Signal) ---
                // Update currentModuleId based on the new structural check
                if (isStructuralModuleWrapper) { 
                    const declPrefix = nodeCodeApprox.substring(0, 50).replace(/\s+/g, '');
                    currentModuleId = `mod_${crypto.createHash('md5').update(declPrefix).digest('hex').slice(0, 6)}`;
                }

                const hasSignal = SIGNAL_KEYWORDS.some(kw => nodeCodeApprox.includes(kw));
                const nodeCategory = hasSignal ? 'priority' : 'vendor';

                const isImport = node.type === 'ImportDeclaration' ||
                    (node.type === 'VariableDeclaration' && nodeCodeApprox.includes('require('));

                // The old isModuleWrapperCall is still used here for smaller, non-recursive wrappers.
                // Re-evaluate its definition if it should also be purely structural, or if it's still good for NPM.
                // For now, let's keep it based on wrapperNames to distinguish if needed.
                const isNamedModuleWrapperCall = wrapperNames.has(node.declarations?.[0]?.init?.callee?.name);

                const isBanal = node.type === 'ExpressionStatement' &&
                    node.expression.type === 'StringLiteral' &&
                    (node.expression.value === 'use strict' || node.expression.value.startsWith('__esModule'));

                const isUtility = node.type === 'VariableDeclaration' &&
                    node.declarations.some(d => d.id.name && (NATIVE_PROPS.includes(d.id.name) || d.id.name.startsWith('_')));

                const isInsideLargeDeclaration = node.type === 'VariableDeclaration' && nodeTokensApprox > 250;

                const shouldSplit = currentChunkNodes.length > 0 && isTopLevelContext && !isInsideLargeDeclaration && (
                    (isImport && currentChunkHasContent) ||
                    (isNamedModuleWrapperCall && currentTokens > 3000) || // Use named for smaller cases
                    (nodeCategory === 'priority' && currentCategory === 'vendor' && currentTokens > 1500) ||
                    (currentTokens + nodeTokensApprox > chunkThreshold) ||
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
        };

        processStatements(statements, true);
        finalizeChunk();
    }

    // Step B: Graph Mapping & Centrality
    detectNeighbors() {
        console.log(`[*] Phase 2: Graph Mapping & Centrality...`);
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
                        const escapedName = definedName.replace(/[.*+?^${}()|[\\]/g, '\\$&');
                        const regex = new RegExp(`\\b${escapedName}\\b`);
                        if (regex.test(nodeData.code)) {
                            nodeData.neighbors.add(targetChunk);
                        }
                    }
                });
            }
        }
    }

    // Phase 2.1: Identifier Affinity (Soft Cohesion Signal)
    calculateAffinity() {
        console.log(`[*] Phase 2.1: Calculating Identifier Affinity (Soft Signal)...`);
        const chunks = Array.from(this.nodes.values()).sort((a, b) => a.id - b.id);

        for (let i = 0; i < chunks.length - 1; i++) {
            const current = chunks[i];
            const next = chunks[i + 1];

            // 1. Module ID Continuity (Hard Signal)
            if (current.moduleId && next.moduleId && current.moduleId === next.moduleId) {
                // already linked by hard signal
                continue;
            }

            // 2. Identifier Cohesion (Soft Signal)
            // Extract defined vars in Current that are used in Next
            const currentDefs = this.extractInternalNames({
                type: 'VariableDeclaration',
                declarations: [] // Structural AST is streamed; rely on regex heuristics below
            });
            // Used a simpler approach: Re-use internalName map logic limited to short names

            // Just scan for short variables defined in A used in B
            const definedInA = new Set();
            const usedInB = new Set();

            const extract = (code, targetSet, isDef) => {
                const ast = parser.parse(code, { sourceType: 'module', plugins: ['jsx', 'typescript'] });
                traverse(ast, {
                    Identifier(p) {
                        if (p.node.name.length <= 2) {
                            if (isDef && (p.isBindingIdentifier() || p.key === p.node)) targetSet.add(p.node.name);
                            else if (!isDef && p.isReferencedIdentifier()) targetSet.add(p.node.name);
                        }
                    }
                });
            };

            try {
                // We can't afford full re-parse performance hit for every chunk pair if it's huge, 
                // but for localized affinity it's okay. Limits:
                if (current.tokens < 5000 && next.tokens < 5000) {
                    // actually, let's use a faster regex heuristic for affinity to be safe
                    const simpleDefRegex = /\b(?:var|let|const|function)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g;
                    let match;
                    while ((match = simpleDefRegex.exec(current.code)) !== null) {
                        if (match[1].length <= 3) definedInA.add(match[1]);
                    }

                    const currentCode = next.code;
                    for (const def of definedInA) {
                        if (currentCode.includes(def)) {
                            // Verify it's not a keyword match with boundary
                            if (new RegExp(`\\b${def}\\b`).test(currentCode)) {
                                usedInB.add(def);
                            }
                        }
                    }
                }
            } catch (e) { continue; }

            const cohesionScore = usedInB.size;
            if (cohesionScore >= 3) {
                console.log(`    [+] Affinity Link: ${current.name} -> ${next.name} (Shared ${cohesionScore} short vars)`);
                current.affinityLink = next.name;
                next.parentChunk = current.name;
                next.affinityScore = cohesionScore;
            }
        }
    }

    // Phase 2.3: Markov Chain Analysis
    applyMarkovAnalysis() {
        console.log(`[*] Phase 2.3: Markov Chain Centrality Calculation (Weighted PageRank variant)...`);
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

    async saveResults(versionDir) {
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
                id: node.id || null,
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
                moduleId: node.moduleId || null,
                affinityLink: node.affinityLink || null,
                parentChunk: node.parentChunk || null,
                affinityScore: node.affinityScore || 0,
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
        const astOutputPath = path.join(metadataDir, 'simplified_asts.json');
        const astStream = fs.createWriteStream(astOutputPath);
        astStream.write('{\n');
        let first = true;

        if (fs.existsSync(this.structuralAstTempPath)) {
            const rl = readline.createInterface({
                input: fs.createReadStream(this.structuralAstTempPath),
                crlfDelay: Infinity
            });
            for await (const line of rl) {
                if (!line) continue;
                let parsed = null;
                try {
                    parsed = JSON.parse(line);
                } catch (err) {
                    console.warn(`[!] Skipping malformed AST line: ${err.message}`);
                    continue;
                }
                const name = parsed.name;
                const node = this.nodes.get(name);
                const payload = {
                    ast: parsed.ast,
                    kb_info: parsed.kb_info,
                    hints: parsed.hints,
                    category: node?.category || parsed.category,
                    role: node?.role || parsed.role,
                    moduleId: node?.moduleId || parsed.moduleId || null
                };
                if (!astStream.write(`${first ? '' : ',\n'}${JSON.stringify(name)}:${JSON.stringify(payload)}`)) {
                    await new Promise(resolve => astStream.once('drain', resolve));
                }
                first = false;
            }
        }

        astStream.write('\n}\n');
        await new Promise(resolve => astStream.end(resolve));
        if (fs.existsSync(this.structuralAstTempPath)) {
            fs.unlinkSync(this.structuralAstTempPath);
        }

        // Optimization: Explicitly clear large temporary structures from the node object 
        // to free up memory before the next bulky operation (Neural Anchoring).
        for (const [, node] of this.nodes) {
            node.code = null;
        }
    }
}


// --- 3. EXECUTION ---
async function run() {
    const { targetFile, version, targetType } = await ensureTargetExists();

    if (!targetFile || !fs.existsSync(targetFile)) {
        console.error(`[!] Error: No input file found.`);
        console.error(`    Checked: CLI arguments, ./cli.js, and ./claude-analysis/*/cli.js (plus binary/cli.js for Bun versions)`);
        console.error(`    Usage: node analyze.js [path-to-bundle.js]`);
        process.exit(1);
    }

    if (targetType === 'binary') {
        console.log(`[*] Native binary downloaded: ${targetFile}`);
        console.warn(`[!] Binary analysis is not supported yet. This is a download-only test.`);
        process.exit(0);
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

    // Determine tempDir
    const tempDir = process.env.GEMINI_TEMP_DIR || '/Users/whit3rabbit/.gemini/tmp/00150962b70d3731010c9badd4003d7d7023e6e06d18dde520c44bbc52c5c1d0'; // Fallback to hardcoded path

    // Write AST inspection log
    fs.writeFileSync(path.join(tempDir, 'ast_inspection_log.jsonl'), '');
    inspectNode(ast, 0, 5, path.join(tempDir, 'ast_inspection_log.jsonl'));
    console.log(`[*] AST inspection log written to ${path.join(tempDir, 'ast_inspection_log.jsonl')}`);

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
    graph.calculateAffinity(); // New Phase 2.5
    graph.detectNeighbors();
    // collapseProxyNodes is already metadata-based
    collapseProxyNodes(graph.nodes);
    graph.applyMarkovAnalysis();
    OUTPUT_BASE = path.join(OUTPUT_ROOT, version);
    console.log(`[*] Output directory: ${OUTPUT_BASE}`);

    await graph.saveResults(OUTPUT_BASE);

    console.log(`\n[COMPLETE] Static Analysis Phase Done.`);
    console.log(`Phase 1-2 results saved to: ${OUTPUT_BASE}`);
    return { version, path: OUTPUT_BASE };
}

module.exports = { simplifyAST };

if (require.main === module) {
    run().catch(err => {
        console.error(`\n[!] FATAL ERROR during analysis:`);
        console.error(err);
        process.exit(1);
    });
}