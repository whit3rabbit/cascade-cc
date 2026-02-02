require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const pLimit = require('p-limit');

const OUTPUT_ROOT = './cascade_graph_analysis';
const { loadKnowledgeBase } = require('./knowledge_base');
const BOOTSTRAP_ROOT = path.join(OUTPUT_ROOT, 'bootstrap');
let KNOWN_PACKAGES = [];
const STRUCTURE_MD_PATH = path.resolve('structrecc.md');
const STRUCTURE_MD_LIMIT = Number.parseInt(process.env.STRUCTURE_MD_LIMIT || '20000', 10);
let STRUCTURE_MD = '';

const { kb: loadedKb, path: kbPath } = loadKnowledgeBase();
if (loadedKb) {
    try {
        const packages = new Set();
        (loadedKb.known_packages || []).forEach(pkg => {
            Object.keys(pkg.dependencies || {}).forEach(dep => packages.add(dep));
            Object.keys(pkg.devDependencies || {}).forEach(dep => packages.add(dep));
        });
        KNOWN_PACKAGES = Array.from(packages).sort();
    } catch (err) {
        const kbLabel = kbPath ? path.basename(kbPath) : 'knowledge_base.json';
        console.warn(`[!] Failed to parse ${kbLabel}: ${err.message}`);
    }
}

if (fs.existsSync(STRUCTURE_MD_PATH)) {
    STRUCTURE_MD = fs.readFileSync(STRUCTURE_MD_PATH, 'utf8');
    if (!Number.isNaN(STRUCTURE_MD_LIMIT) && STRUCTURE_MD_LIMIT > 0 && STRUCTURE_MD.length > STRUCTURE_MD_LIMIT) {
        STRUCTURE_MD = `${STRUCTURE_MD.slice(0, STRUCTURE_MD_LIMIT)}\n\n<!-- truncated -->\n`;
    }
}

function toPosixPath(p) {
    return p.split(path.sep).join('/');
}

function normalizeRelPath(relPath) {
    const normalized = path.posix.normalize(relPath).replace(/^(\.\.(\/|\\|$))+/, '');
    return normalized.replace(/^\/+/, '');
}

function resolveRefinedPath(refinedRoot, relPath) {
    const safeRel = normalizeRelPath(toPosixPath(relPath));
    return path.join(refinedRoot, safeRel);
}

function buildInternalIndex(rootDir) {
    const files = [];
    function walk(dir) {
        fs.readdirSync(dir).forEach(file => {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                walk(fullPath);
            } else if (file.endsWith('.js')) {
                files.push(toPosixPath(path.relative(rootDir, fullPath)));
            }
        });
    }
    walk(rootDir);
    files.sort();
    return files;
}

function stripExtension(p) {
    return p.replace(/\.[^.]+$/, '');
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
    const structSim = aStruct.reduce((sum, a, i) => sum + a * bStruct[i], 0);
    if (aLit && bLit) {
        const litSim = aLit.reduce((sum, a, i) => sum + a * bLit[i], 0);
        return (structWeight * structSim) + (litWeight * litSim);
    }
    return structSim;
}

function extractLibFromLabel(label) {
    if (!label) return null;
    const idx = label.indexOf('_');
    return idx === -1 ? null : label.slice(0, idx);
}

function loadBootstrapSource(libName) {
    if (!libName) return '';
    const bootstrapRoot = path.join(__dirname, '..', 'ml', 'bootstrap_data');
    if (!fs.existsSync(bootstrapRoot)) return '';

    const candidates = fs.readdirSync(bootstrapRoot)
        .map(name => path.join(bootstrapRoot, name))
        .filter(p => fs.statSync(p).isDirectory())
        .filter(p => {
            const base = path.basename(p);
            return base === libName || base.startsWith(`${libName}_`) || base.startsWith(`_${libName}`);
        })
        .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);

    if (candidates.length === 0) return '';
    const dir = candidates[0];
    const entryPath = path.join(dir, 'entry.js');
    if (fs.existsSync(entryPath)) return fs.readFileSync(entryPath, 'utf8');
    const bundledPath = path.join(dir, 'bundled.js');
    if (fs.existsSync(bundledPath)) return fs.readFileSync(bundledPath, 'utf8');
    return '';
}

function getLatestBootstrapDir(rootDir) {
    if (!fs.existsSync(rootDir)) return null;
    const entries = fs.readdirSync(rootDir)
        .map(name => path.join(rootDir, name))
        .filter(p => fs.statSync(p).isDirectory());
    if (entries.length === 0) return null;
    entries.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
    return entries[0];
}

function getLatestCustomGoldDir(rootDir) {
    if (!fs.existsSync(rootDir)) return null;
    const entries = fs.readdirSync(rootDir)
        .map(name => path.join(rootDir, name))
        .filter(p => fs.statSync(p).isDirectory())
        .filter(p => {
            const base = path.basename(p);
            if (!base.startsWith('custom_claude_gold_v')) return false;
            if (base.includes('esbuild') || base.includes('bun')) return false;
            return true;
        });
    if (entries.length === 0) return null;
    const parseVersion = dirPath => {
        const base = path.basename(dirPath);
        const match = base.match(/^custom_claude_gold_v(\d+(?:_\d+)*)/);
        if (!match) return [];
        return match[1].split('_').map(part => Number.parseInt(part, 10)).filter(n => Number.isFinite(n));
    };
    const compareVersions = (a, b) => {
        const va = parseVersion(a);
        const vb = parseVersion(b);
        const len = Math.max(va.length, vb.length);
        for (let i = 0; i < len; i += 1) {
            const na = va[i] ?? 0;
            const nb = vb[i] ?? 0;
            if (na !== nb) return nb - na;
        }
        return 0;
    };
    entries.sort((a, b) => {
        const cmp = compareVersions(a, b);
        if (cmp !== 0) return cmp;
        return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs;
    });
    return entries[0];
}

function buildGoldIndex(rootDir) {
    const files = [];
    function walk(dir) {
        fs.readdirSync(dir).forEach(file => {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                walk(fullPath);
            } else if (file.endsWith('.js') || file.endsWith('.ts')) {
                files.push(fullPath);
            }
        });
    }
    walk(rootDir);

    const byPath = new Map();
    const byBase = new Map();
    files.forEach(fullPath => {
        const relPath = toPosixPath(path.relative(rootDir, fullPath));
        const relNoExt = stripExtension(relPath);
        byPath.set(relNoExt, fullPath);
        const base = path.posix.basename(relNoExt);
        if (!byBase.has(base)) byBase.set(base, []);
        byBase.get(base).push(fullPath);
    });

    return { rootDir, byPath, byBase };
}

function resolveGoldMatch(relPathPosix, goldIndex) {
    if (!goldIndex) return null;
    const relNoExt = stripExtension(relPathPosix);
    if (goldIndex.byPath.has(relNoExt)) {
        return goldIndex.byPath.get(relNoExt);
    }
    const base = path.posix.basename(relNoExt);
    const matches = goldIndex.byBase.get(base);
    if (matches && matches.length === 1) return matches[0];
    return null;
}

function truncateGold(code, maxChars = 8000) {
    if (!code) return '';
    if (code.length <= maxChars) return code;
    return `${code.slice(0, maxChars)}\n/* ...truncated gold reference... */`;
}

function buildPathIndex(internalIndex) {
    const byBase = new Map();
    internalIndex.forEach(p => {
        const base = path.posix.basename(p, '.js');
        if (!byBase.has(base)) byBase.set(base, []);
        byBase.get(base).push(p);
    });
    return { byBase };
}

function extractImportSpecifiers(code) {
    const specs = new Set();
    const importFromRe = /import\s+[^'"]*?from\s+['"]([^'"]+)['"]/g;
    const importBareRe = /import\s+['"]([^'"]+)['"]/g;
    const requireRe = /require\(\s*['"]([^'"]+)['"]\s*\)/g;
    const dynamicImportRe = /import\(\s*['"]([^'"]+)['"]\s*\)/g;

    let m;
    while ((m = importFromRe.exec(code))) specs.add(m[1]);
    while ((m = importBareRe.exec(code))) specs.add(m[1]);
    while ((m = requireRe.exec(code))) specs.add(m[1]);
    while ((m = dynamicImportRe.exec(code))) specs.add(m[1]);

    return Array.from(specs);
}

function resolveImportSpecifier(spec, relDir, internalIndexSet, pathIndex) {
    const normalize = p => p.replace(/\\/g, '/');
    const withJs = p => (p.endsWith('.js') ? p : `${p}.js`);

    if (spec.startsWith('.')) {
        const candidate = normalize(path.posix.normalize(path.posix.join(relDir, spec)));
        const candidateJs = withJs(candidate);
        if (internalIndexSet.has(candidateJs)) return candidateJs;
        if (internalIndexSet.has(candidate)) return candidate;
        return null;
    }

    if (spec.startsWith('/')) {
        const candidate = normalize(spec.replace(/^\//, ''));
        const candidateJs = withJs(candidate);
        if (internalIndexSet.has(candidateJs)) return candidateJs;
        if (internalIndexSet.has(candidate)) return candidate;
        return null;
    }

    const lastSegment = normalize(spec).split('/').pop();
    if (lastSegment && pathIndex.byBase.has(lastSegment)) {
        const matches = pathIndex.byBase.get(lastSegment);
        if (matches.length === 1) return matches[0];
    }

    return null;
}

function isKnownPackage(spec, knownPackages) {
    if (knownPackages.includes(spec)) return true;
    const parts = spec.split('/');
    if (spec.startsWith('@') && parts.length >= 2) {
        const scopePkg = `${parts[0]}/${parts[1]}`;
        if (knownPackages.includes(scopePkg)) return true;
    }
    const root = parts[0];
    return knownPackages.includes(root);
}

const parsedRefineLimit = process.env.REFINE_CONTEXT_LIMIT !== undefined
    ? Number.parseInt(process.env.REFINE_CONTEXT_LIMIT, 10)
    : NaN;
const MAX_REFINE_CHARS = Number.isNaN(parsedRefineLimit) ? 400000 : parsedRefineLimit;
const STATUS_FILE = 'refine_status.json';
const CUSTOM_PACKAGE_JSON_BASE = {
    name: 'claude-code-deobfuscated',
    version: '0.0.0',
    description: 'Best attempt deobfuscation of the command-line interface for Claude.',
    type: 'module',
    main: 'dist/entrypoints/cli.js',
    bin: {
        claude: 'cli.js',
    },
    files: [
        'dist',
        'vendor',
        'schemas',
    ],
    scripts: {
        dev: 'tsx src/entrypoints/cli.tsx',
        'dev:bun': 'bun run dev',
        build: 'node node_modules/typescript/bin/tsc',
        'build:bun': 'bun run build',
        bundle: "esbuild src/entrypoints/cli.tsx --bundle --platform=node --target=node18 --format=esm --outfile=cli.js --define:process.env.NODE_ENV='\"production\"' --external:fsevents --external:tree-sitter --external:tree-sitter-typescript --external:sharp --external:yoga-layout-prebuilt --alias:react-devtools-core=./src/vendor/react-devtools-core-stub.ts --banner:js=\"#!/usr/bin/env node\nimport { createRequire as _createRequire } from 'module'; const require = _createRequire(import.meta.url);\"",
        package: 'npm run bundle && pkg . --targets node18-linux-x64,node18-macos-x64,node18-win-x64 --out-path binaries',
        dist: 'tsx scripts/build-dist.ts',
        'generate:schemas': 'tsx scripts/generate-tool-schemas.ts',
        'generate:types': 'tsx scripts/generate-types.ts',
        prebuild: 'npm run generate:types',
        lint: 'eslint . --ext .ts,.tsx',
        test: 'NODE_OPTIONS=--experimental-vm-modules jest',
        'test:bun': 'bun run test',
        prepublishOnly: 'npm run build && npm run bundle',
        clean: 'rm -rf dist binaries',
    },
    dependencies: {
        '@anthropic-ai/bedrock-sdk': '^0.26.0',
        '@anthropic-ai/sdk': '^0.71.2',
        '@anthropic-ai/vertex-sdk': '^0.14.0',
        '@aws-sdk/client-bedrock': '^3.962.0',
        '@aws-sdk/client-bedrock-runtime': '^3.962.0',
        '@aws-sdk/client-s3': '^3.958.0',
        '@aws-sdk/client-sts': '^3.958.0',
        '@aws-sdk/credential-providers': '^3.958.0',
        '@azure/msal-node': '^2.6.0',
        '@inkjs/ui': '^2.0.0',
        '@modelcontextprotocol/sdk': '^1.25.1',
        '@opentelemetry/api': '^1.9.0',
        '@opentelemetry/auto-instrumentations-node': '^0.67.3',
        '@opentelemetry/core': '^2.2.0',
        '@opentelemetry/exporter-logs-otlp-http': '^0.208.0',
        '@opentelemetry/exporter-metrics-otlp-http': '^0.208.0',
        '@opentelemetry/exporter-trace-otlp-http': '^0.208.0',
        '@opentelemetry/resources': '^2.2.0',
        '@opentelemetry/sdk-logs': '^0.208.0',
        '@opentelemetry/sdk-metrics': '^2.2.0',
        '@opentelemetry/sdk-trace-node': '^2.2.0',
        '@opentelemetry/semantic-conventions': '^1.38.0',
        '@resvg/resvg-js': '^2.6.2',
        '@resvg/resvg-wasm': '^2.6.2',
        '@segment/analytics-node': '*',
        '@sentry/node': '^10.32.1',
        'abort-controller': '^3.0.0',
        ajv: '^8.17.1',
        'ansi-escapes': '^7.2.0',
        'ansi-styles': '^6.2.3',
        axios: '^1.13.2',
        chalk: '^5.6.2',
        chokidar: '^5.0.0',
        'cli-highlight': '^2.1.11',
        'cli-table3': '^0.6.5',
        commander: '^14.0.2',
        'date-fns': '*',
        diff: '^8.0.2',
        domino: '*',
        dotenv: '^17.2.3',
        'error-stack-parser': '*',
        execa: '^9.6.1',
        fflate: '*',
        figures: '^6.1.0',
        'fuse.js': '^7.1.0',
        'grapheme-splitter': '^1.0.4',
        'gray-matter': '^4.0.3',
        'highlight.js': '^11.11.1',
        'html-entities': '*',
        'https-proxy-agent': '^7.0.6',
        ink: '^6.6.0',
        'ink-link': '^5.0.0',
        'ink-select-input': '^6.2.0',
        'ink-spinner': '^5.0.0',
        'ink-text-input': '^6.0.0',
        'is-unicode-supported': '^2.1.0',
        'js-yaml': '^4.1.1',
        jsdom: '^27.4.0',
        json5: '^2.2.3',
        localforage: '*',
        'lodash-es': '^4.17.22',
        'lru-cache': '^11.2.4',
        marked: '^17.0.1',
        memoize: '^10.2.0',
        micromatch: '^4.0.8',
        'mime-types': '^3.0.2',
        'node-pty': '^1.1.0',
        open: '^11.0.0',
        'ordered-map': '^0.1.0',
        parse5: '*',
        'pdf-parse': '^2.4.5',
        plist: '^3.1.0',
        'proper-lockfile': '^4.1.2',
        react: '^19.2.3',
        semver: '^7.7.3',
        sharp: '^0.34.5',
        'shell-quote': '^1.8.3',
        'statsig-js': '^5.1.0',
        'string-width': '^8.1.0',
        'supports-hyperlinks': '^4.4.0',
        'tree-sitter': '^0.21.0',
        'tree-sitter-bash': '^0.25.1',
        'tree-sitter-typescript': '^0.23.2',
        tslib: '*',
        turndown: '^7.2.2',
        undici: '^7.19.2',
        'uri-js': '*',
        uuid: '*',
        'vscode-ripgrep': '^1.13.2',
        wcwidth: '^1.0.1',
        'web-tree-sitter': '^0.26.3',
        'word-wrap': '^1.2.5',
        'wrap-ansi': '^9.0.2',
        ws: '^8.18.3',
        xmlbuilder2: '^4.0.3',
        xss: '*',
        'yoga-layout-prebuilt': '^1.10.0',
        zod: '^4.2.1',
    },
    devDependencies: {
        '@types/jest': '^30.0.0',
        '@types/js-yaml': '^4.0.9',
        '@types/jsdom': '^27.0.0',
        '@types/lodash-es': '^4.17.12',
        '@types/node': '^25.0.3',
        '@types/orderedmap': '^2.0.0',
        '@types/plist': '^3.0.5',
        '@types/proper-lockfile': '^4.1.4',
        '@types/react': '^19.2.7',
        '@types/semver': '^7.7.1',
        '@types/shell-quote': '^1.7.5',
        '@types/turndown': '^5.0.6',
        '@types/ws': '^8.18.1',
        '@typescript-eslint/eslint-plugin': '^8.51.0',
        '@typescript-eslint/parser': '^8.51.0',
        esbuild: '^0.27.2',
        eslint: '^9.39.2',
        jest: '^30.2.0',
        'json-schema-to-typescript': '^15.0.4',
        pkg: '^5.8.1',
        'ts-jest': '^29.4.6',
        tsx: '^4.21.0',
        typescript: '^5.9.3',
        'zod-to-json-schema': '^3.25.1',
    },
    engines: {
        node: '>=18',
        bun: '>=1.0.0',
    },
    packageManager: 'bun@1.0.0',
    pkg: {
        assets: [
            'assets/**/*',
            'vendor/**/*',
            'schemas/**/*',
            'cli.js',
        ],
        outputPath: 'binaries',
    },
    author: 'Anthropic, PBC',
    license: 'Apache-2.0',
};

function buildCustomPackageJson(version) {
    const payload = {
        ...CUSTOM_PACKAGE_JSON_BASE,
        version,
    };
    return JSON.stringify(payload, null, 2);
}

function loadStatus(statusPath) {
    if (!fs.existsSync(statusPath)) {
        return { files: {}, updatedAt: null, limitChars: MAX_REFINE_CHARS };
    }
    try {
        const parsed = JSON.parse(fs.readFileSync(statusPath, 'utf8'));
        if (!parsed || typeof parsed !== 'object') return { files: {} };
        if (!parsed.files || typeof parsed.files !== 'object') parsed.files = {};
        return parsed;
    } catch (err) {
        console.warn(`[!] Failed to read status file ${statusPath}: ${err.message}`);
        return { files: {}, updatedAt: null, limitChars: MAX_REFINE_CHARS };
    }
}

function createStatusWriter(statusPath, status) {
    let writeChain = Promise.resolve();
    return function persistStatus() {
        const snapshot = {
            ...status,
            updatedAt: new Date().toISOString(),
            limitChars: MAX_REFINE_CHARS,
        };
        writeChain = writeChain.then(() => {
            fs.writeFileSync(statusPath, JSON.stringify(snapshot, null, 2));
        });
        return writeChain;
    };
}

function updateStatus(status, relPath, entry) {
    status.files[relPath] = {
        ...entry,
        updatedAt: new Date().toISOString(),
    };
}

function stripFence(text) {
    if (!text) return '';
    let cleaned = text.trim();
    if (cleaned.startsWith('```json')) {
        cleaned = cleaned.replace(/^```json\n/, '').replace(/\n```$/, '');
    } else if (cleaned.startsWith('```javascript')) {
        cleaned = cleaned.replace(/^```javascript\n/, '').replace(/\n```$/, '');
    } else if (cleaned.startsWith('```')) {
        cleaned = cleaned.replace(/^```\n/, '').replace(/\n```$/, '');
    }
    return cleaned.trim();
}

function validateRefineOps(ops) {
    if (!ops || typeof ops !== 'object') {
        throw new Error('Invalid refine response: expected JSON object.');
    }
    const files = Array.isArray(ops.files) ? ops.files : [];
    const create = Array.isArray(ops.create) ? ops.create : [];
    const update = Array.isArray(ops.update) ? ops.update : [];
    const del = Array.isArray(ops.delete) ? ops.delete : [];
    const hasWrites = files.length > 0 || create.length > 0 || update.length > 0;
    if (!hasWrites) {
        throw new Error('Invalid refine response: no files to write.');
    }
    const allWrites = files.concat(create, update);
    allWrites.forEach((entry, idx) => {
        if (!entry || typeof entry !== 'object') {
            throw new Error(`Invalid refine response: write entry ${idx} is not an object.`);
        }
        if (typeof entry.path !== 'string' || entry.path.trim() === '') {
            throw new Error(`Invalid refine response: write entry ${idx} missing path.`);
        }
        if (typeof entry.code !== 'string') {
            throw new Error(`Invalid refine response: write entry ${idx} missing code.`);
        }
        const safe = normalizeRelPath(toPosixPath(entry.path));
        if (!safe || safe.startsWith('..')) {
            throw new Error(`Invalid refine response: write entry ${idx} has invalid path.`);
        }
    });
    del.forEach((entry, idx) => {
        if (typeof entry !== 'string' || entry.trim() === '') {
            throw new Error(`Invalid refine response: delete entry ${idx} invalid.`);
        }
        const safe = normalizeRelPath(toPosixPath(entry));
        if (!safe || safe.startsWith('..')) {
            throw new Error(`Invalid refine response: delete entry ${idx} has invalid path.`);
        }
    });
    return { files, create, update, delete: del };
}

function parseRefineResponse(raw) {
    const cleaned = stripFence(raw);
    if (!cleaned) {
        throw new Error('Invalid refine response: empty response.');
    }
    if (!(cleaned.startsWith('{') || cleaned.startsWith('['))) {
        throw new Error('Invalid refine response: expected JSON.');
    }
    const parsed = JSON.parse(cleaned);
    const ops = validateRefineOps(parsed);
    return { type: 'ops', ops };
}

async function refineFile(filePath, relPath, refinedRoot, internalIndex, pathIndex, goldIndex, customGoldIndex, graphChunks, logicByName, logicRegistry, status, persistStatus, packageJsonText) {
    const code = fs.readFileSync(filePath, 'utf8');

    // Skip very large files or vendor files if needed, but for now let's try all
    if (MAX_REFINE_CHARS > 0 && code.length > MAX_REFINE_CHARS) {
        console.warn(`[!] Skipping ${relPath} (too large: ${code.length} chars, limit: ${MAX_REFINE_CHARS})`);
        updateStatus(status, relPath, {
            status: 'skipped',
            reason: 'too_large',
            sizeChars: code.length,
            limitChars: MAX_REFINE_CHARS,
        });
        await persistStatus();
        return;
    }

    const relPathPosix = toPosixPath(relPath);
    const relDir = path.posix.dirname(relPathPosix);
    const localPaths = internalIndex.filter(p => p.startsWith(`${relDir}/`)).slice(0, 120);
    const otherPaths = internalIndex.filter(p => !p.startsWith(`${relDir}/`)).slice(0, 120);
    const internalIndexSet = new Set(internalIndex);

    const importSpecs = extractImportSpecifiers(code);
    const importHints = [];
    const unresolvedBares = [];
    importSpecs.forEach(spec => {
        if (spec.startsWith('.') || spec.startsWith('/')) {
            const resolved = resolveImportSpecifier(spec, relDir, internalIndexSet, pathIndex);
            if (resolved) {
                const relativeHint = resolved.startsWith(relDir + '/')
                    ? `./${resolved.slice(relDir.length + 1)}`
                    : path.posix.relative(relDir, resolved).startsWith('.')
                        ? path.posix.relative(relDir, resolved)
                        : `./${path.posix.relative(relDir, resolved)}`;
                importHints.push({ spec, suggestion: relativeHint, reason: 'internal path match' });
            }
        } else {
            if (!isKnownPackage(spec, KNOWN_PACKAGES)) {
                const resolved = resolveImportSpecifier(spec, relDir, internalIndexSet, pathIndex);
                if (resolved) {
                    const relativeHint = resolved.startsWith(relDir + '/')
                        ? `./${resolved.slice(relDir.length + 1)}`
                        : path.posix.relative(relDir, resolved).startsWith('.')
                            ? path.posix.relative(relDir, resolved)
                            : `./${path.posix.relative(relDir, resolved)}`;
                    importHints.push({ spec, suggestion: relativeHint, reason: 'matched internal file basename' });
                } else {
                    unresolvedBares.push(spec);
                }
            }
        }
    });

    const goldMatchPath = resolveGoldMatch(relPathPosix, goldIndex);
    let goldReference = '';
    if (goldMatchPath) {
        try {
            goldReference = truncateGold(fs.readFileSync(goldMatchPath, 'utf8'));
        } catch (err) {
            console.warn(`[!] Failed reading gold match for ${relPathPosix}: ${err.message}`);
        }
    }

    const customGoldMatchPath = resolveGoldMatch(relPathPosix, customGoldIndex);
    let customGoldReference = '';
    if (customGoldMatchPath) {
        try {
            customGoldReference = truncateGold(fs.readFileSync(customGoldMatchPath, 'utf8'));
        } catch (err) {
            console.warn(`[!] Failed reading custom gold match for ${relPathPosix}: ${err.message}`);
        }
    }

    const relNoExt = stripExtension(relPathPosix);
    const matchedChunks = (graphChunks || []).filter(chunk => {
        const candidates = [
            chunk.proposedPath,
            chunk.suggestedPath,
            chunk.kb_info?.suggested_path
        ].filter(Boolean).map(p => stripExtension(toPosixPath(p)));
        return candidates.includes(relNoExt);
    });

    let structuralReference = '';
    let structuralReferenceSimilarity = null;
    if (logicRegistry && matchedChunks.length > 0) {
        const founderChunk = matchedChunks.find(c => c.category === 'founder') ||
            matchedChunks.find(c => c.role === 'ENTRY_POINT') ||
            matchedChunks.slice().sort((a, b) => (b.centrality || 0) - (a.centrality || 0))[0];

        if (founderChunk) {
            const logicEntry = logicByName.get(founderChunk.name);
            const chunkVectors = resolveVectors(logicEntry);
            if (chunkVectors.structural) {
                let bestMatch = { label: null, ref: null, similarity: -1 };
                for (const [label, refData] of Object.entries(logicRegistry)) {
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
                if (bestMatch.similarity >= 0.8) {
                    const libName = bestMatch.ref?.lib || extractLibFromLabel(bestMatch.label);
                    const source = loadBootstrapSource(libName);
                    if (source) {
                        structuralReferenceSimilarity = bestMatch.similarity;
                        structuralReference = truncateGold(source);
                    }
                }
            }
        }
    }

    const ignoreLibs = new Set();
    matchedChunks.forEach(chunk => {
        const sim = chunk.matchSimilarityBoosted ?? chunk.matchSimilarity ?? 0;
        const label = chunk.matchLabel;
        if (sim >= 0.95 && label) {
            const lib = extractLibFromLabel(label);
            if (lib) ignoreLibs.add(lib);
        }
    });

    const basePrompt = `
Role: Senior Staff Software Engineer / Reverse Engineer
Task: Source Code Reconstruction & Logic Refinement

I have an assembled JavaScript file that was deobfuscated from a minified bundle.
The identifiers names are mostly correct, but the logic structure might still be "minified" (e.g., flattened loops, complex ternary chains, inlined constants, dead code branches).

GOAL: Reconstruct this file into what the ORIGINAL source code likely looked like.

${goldReference ? `GOLD REFERENCE (Latest Bootstrap Match):
Use this file as a high-confidence reference for naming and structure when it clearly aligns.
Do not copy unless it matches the assembled file's logic.

\`\`\`javascript
${goldReference}
\`\`\`
` : ''}

${customGoldReference ? `CUSTOM GOLD REFERENCE (Latest Claude Gold, Use as Primary Guide):
This is the deobfuscated reference for the equivalent file. Match its structure and naming closely,
but preserve any new or divergent logic from the assembled file.

\`\`\`javascript
${customGoldReference}
\`\`\`
` : ''}

${structuralReference ? `STRUCTURAL_REFERENCE (Neural Registry Match, ${(structuralReferenceSimilarity * 100).toFixed(2)}%):
Use this as a structural guide only when it aligns closely; do not force matches that diverge.

\`\`\`javascript
${structuralReference}
\`\`\`
` : ''}

${ignoreLibs.size > 0 ? `IGNORE LIST:
These chunks are confirmed vendor libraries. Do NOT rename internal variables to proprietary names.
${Array.from(ignoreLibs).map(lib => `- ${lib.toUpperCase()}`).join('\n')}
` : ''}

PROJECT STRUCTURE REFERENCE (structrecc.md):
${STRUCTURE_MD || 'Structure not available.'}

PACKAGE.JSON CONTEXT:
\`\`\`json
${packageJsonText}
\`\`\`

IMPORT RESOLUTION CONTEXT:
- Known internal module paths (same folder): ${localPaths.length > 0 ? `\n${localPaths.map(p => `  - ${p}`).join('\n')}` : 'None'}
- Known internal module paths (other folders, sample): ${otherPaths.length > 0 ? `\n${otherPaths.slice(0, 120).map(p => `  - ${p}`).join('\n')}` : 'None'}
- Known external packages: ${KNOWN_PACKAGES.length > 0 ? `\n${KNOWN_PACKAGES.map(p => `  - ${p}`).join('\n')}` : 'None'}
- Import resolution hints for this file: ${importHints.length > 0 ? `\n${importHints.slice(0, 60).map(h => `  - ${h.spec} -> ${h.suggestion} (${h.reason})`).join('\n')}` : 'None'}
- Unrecognized bare imports (do not invent packages): ${unresolvedBares.length > 0 ? `\n${unresolvedBares.slice(0, 60).map(s => `  - ${s}`).join('\n')}` : 'None'}

INSTRUCTIONS:
0. Use the custom_knowledge_base.json project structure as the PRIMARY guide: prioritize existing folders and nearby module paths, but allow creating a new file path when the logic genuinely doesn't fit any known path.
1. Restore clean control flow (use if/else instead of complex ternaries where appropriate).
2. Group related functions/variables logically.
3. Remove any remaining obfuscation artifacts (like proxy functions or unused helper calls).
4. Ensure the exports match a modern ESM/CommonJS structure.
5. Fix import specifiers so they resolve:
   - If an import matches a known internal module, rewrite it to the correct relative path.
   - If an import is a bare specifier, only use packages from the Known external packages list.
   - If an import could plausibly be a vendor library, prefer the vendor package import (from Known external packages) over creating or referencing internal vendor code.
   - Do not invent new package names. If unsure, keep the original specifier.
6. Vendor libraries (e.g. zod/react) were filtered earlier. Do not create internal files or import paths named after vendor libraries (e.g. ./zod, ./react, src/zod/*).
7. Some vendor code may still be embedded in the file: do NOT inline vendor library code. Use proper package imports instead.
8. If the file seems to contain multiple logical modules, split it into multiple files and adjust imports/exports accordingly.
9. If the logic clearly belongs in a new file or folder, create it and update imports/exports to match.
10. Add helpful comments explaining complex logic blocks.
11. FIX any obviously broken logic caused by the assembly process (e.g. out-of-order definitions if detected).

FILE PATH: ${relPathPosix}

CODE:
\`\`\`javascript
${code}
\`\`\`

RESPONSE (JSON ONLY):
Return a JSON object that follows this schema. Do NOT include markdown or any other text.
Schema:
{
  "files": [ { "path": "relative/path.js", "code": "..." } ],
  "create": [ { "path": "relative/new.js", "code": "..." } ],
  "update": [ { "path": "relative/existing.js", "code": "..." } ],
  "delete": [ "relative/old.js" ]
}
Rules:
- Include at least one entry in files/create/update.
- Use POSIX-style paths relative to the project root.
- Only include delete when needed.
- JSON must be valid; all code must be a JSON string with proper escaping (e.g. \\n, \\t, \\\"), not raw multiline text.

Example:
{
  "files": [
    { "path": "relative/path.js", "code": "..." }
  ],
  "delete": ["relative/old.js"]
}
`;

    console.log(`[*] Refining ${relPath}...`);
    try {
        const maxAttempts = 3;
        let parsed = null;
        let lastError = null;
        for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
            const retryNote = attempt === 1
                ? ''
                : `\n\nIMPORTANT: Your previous response was invalid JSON for this schema.\nError: ${lastError}\nReturn ONLY a valid JSON object that matches the schema.`;
            const prompt = `${basePrompt}${retryNote}`;
            const refinedCode = await callLLM(prompt);
            try {
                parsed = parseRefineResponse(refinedCode);
                lastError = null;
                break;
            } catch (err) {
                lastError = err.message || String(err);
                parsed = null;
            }
        }
        if (!parsed) {
            throw new Error(`Refine JSON validation failed after ${maxAttempts} attempts: ${lastError || 'unknown error'}`);
        }
        const written = [];
        const deleted = [];
        const ops = parsed.ops || {};
        const combined = ops.files.concat(ops.create, ops.update);
        combined.forEach(entry => {
            const refinedPath = resolveRefinedPath(refinedRoot, entry.path);
            const refinedDir = path.dirname(refinedPath);
            if (!fs.existsSync(refinedDir)) fs.mkdirSync(refinedDir, { recursive: true });
            fs.writeFileSync(refinedPath, entry.code);
            written.push(toPosixPath(path.relative(OUTPUT_ROOT, refinedPath)));
            console.log(`[+] Saved refined file: ${refinedPath}`);
        });
        ops.delete.forEach(rel => {
            const refinedPath = resolveRefinedPath(refinedRoot, rel);
            if (fs.existsSync(refinedPath)) {
                fs.unlinkSync(refinedPath);
                deleted.push(toPosixPath(path.relative(OUTPUT_ROOT, refinedPath)));
                console.log(`[+] Deleted refined file: ${refinedPath}`);
            }
        });

        updateStatus(status, relPath, {
            status: 'done',
            sizeChars: code.length,
            refinedPath: written,
            deleted,
            responseType: parsed.type,
        });
        await persistStatus();
    } catch (err) {
        console.error(`[!] Error refining ${relPath}: ${err.message}`);
        updateStatus(status, relPath, {
            status: 'error',
            reason: err.message,
            sizeChars: code.length,
        });
        await persistStatus();
    }
}

async function run() {
    let version = process.argv[2];
    if (!version) {
        console.error("Usage: node src/refine_codebase.js <version>");
        process.exit(1);
    }

    const assembleDir = path.join(OUTPUT_ROOT, version, 'assemble');
    if (!fs.existsSync(assembleDir)) {
        console.error(`[!] Assemble directory not found: ${assembleDir}`);
        process.exit(1);
    }
    const refinedRoot = path.join(OUTPUT_ROOT, version, 'refined_assemble');
    if (!fs.existsSync(refinedRoot)) fs.mkdirSync(refinedRoot, { recursive: true });
    const statusPath = path.join(refinedRoot, STATUS_FILE);
    const status = loadStatus(statusPath);
    const persistStatus = createStatusWriter(statusPath, status);

    const isValid = await validateKey();
    if (!isValid) process.exit(1);

    const files = [];
    function walk(dir) {
        fs.readdirSync(dir).forEach(file => {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                walk(fullPath);
            } else if (file.endsWith('.js')) {
                files.push(fullPath);
            }
        });
    }
    walk(assembleDir);

    console.log(`[*] Found ${files.length} files to refine.`);

    const internalIndex = buildInternalIndex(assembleDir);

    const pathIndex = buildPathIndex(internalIndex);
    const graphMapPath = path.join(OUTPUT_ROOT, version, 'metadata', 'graph_map.json');
    const graphDataRaw = fs.existsSync(graphMapPath) ? JSON.parse(fs.readFileSync(graphMapPath, 'utf8')) : [];
    const graphChunks = Array.isArray(graphDataRaw) ? graphDataRaw : (graphDataRaw.chunks || []);

    const logicDbPath = path.join(OUTPUT_ROOT, version, 'metadata', 'logic_db.json');
    const logicDb = fs.existsSync(logicDbPath) ? JSON.parse(fs.readFileSync(logicDbPath, 'utf8')) : [];
    const logicByName = new Map(logicDb.map(entry => [entry.name, entry]));

    const registryPath = path.join(OUTPUT_ROOT, 'logic_registry.json');
    const logicRegistry = fs.existsSync(registryPath) ? JSON.parse(fs.readFileSync(registryPath, 'utf8')) : null;
    const latestBootstrapDir = getLatestBootstrapDir(BOOTSTRAP_ROOT);
    const goldIndex = latestBootstrapDir ? buildGoldIndex(latestBootstrapDir) : null;
    if (latestBootstrapDir) {
        console.log(`[*] Using latest bootstrap gold: ${toPosixPath(path.relative(OUTPUT_ROOT, latestBootstrapDir))}`);
    } else {
        console.warn(`[!] No bootstrap gold directory found at ${BOOTSTRAP_ROOT}`);
    }
    const customGoldDir = getLatestCustomGoldDir(BOOTSTRAP_ROOT);
    const customGoldIndex = customGoldDir ? buildGoldIndex(customGoldDir) : null;
    if (customGoldDir) {
        console.log(`[*] Using latest custom gold: ${toPosixPath(path.relative(OUTPUT_ROOT, customGoldDir))}`);
    } else {
        console.warn(`[!] No custom gold directory found under ${BOOTSTRAP_ROOT}`);
    }
    const limit = pLimit(PROVIDER === 'gemini' ? 2 : 2); // Conservative limit for all providers to avoid timeouts
    const packageJsonText = buildCustomPackageJson(version);
    const tasks = files.map(file => {
        const relPath = path.relative(assembleDir, file);
        const refinedPath = path.join(refinedRoot, relPath);
        const existing = status.files[relPath];
        if (fs.existsSync(refinedPath)) {
            if (!existing || existing.status !== 'done') {
                updateStatus(status, relPath, {
                    status: 'done',
                    reason: 'existing_refined',
                    refinedPath: toPosixPath(path.relative(OUTPUT_ROOT, refinedPath)),
                });
                persistStatus();
            }
            return null;
        }
        if (existing && existing.status === 'done') {
            return null;
        }
        return limit(() => refineFile(file, relPath, refinedRoot, internalIndex, pathIndex, goldIndex, customGoldIndex, graphChunks, logicByName, logicRegistry, status, persistStatus, packageJsonText));
    }).filter(Boolean);

    await Promise.all(tasks);
    await persistStatus();
    console.log(`[*] Refinement complete.`);
}

run().catch(console.error);
