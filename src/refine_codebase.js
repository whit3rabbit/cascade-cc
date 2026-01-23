require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const pLimit = require('p-limit');

const OUTPUT_ROOT = './cascade_graph_analysis';
const KB_PATH = './knowledge_base.json';
const BOOTSTRAP_ROOT = path.join(OUTPUT_ROOT, 'bootstrap');
let KNOWN_PACKAGES = [];

if (fs.existsSync(KB_PATH)) {
    try {
        const kb = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
        const packages = new Set();
        (kb.known_packages || []).forEach(pkg => {
            Object.keys(pkg.dependencies || {}).forEach(dep => packages.add(dep));
            Object.keys(pkg.devDependencies || {}).forEach(dep => packages.add(dep));
        });
        KNOWN_PACKAGES = Array.from(packages).sort();
    } catch (err) {
        console.warn(`[!] Failed to parse ${KB_PATH}: ${err.message}`);
    }
}

function toPosixPath(p) {
    return p.split(path.sep).join('/');
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

function getLatestBootstrapDir(rootDir) {
    if (!fs.existsSync(rootDir)) return null;
    const entries = fs.readdirSync(rootDir)
        .map(name => path.join(rootDir, name))
        .filter(p => fs.statSync(p).isDirectory());
    if (entries.length === 0) return null;
    entries.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
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

const MAX_REFINE_CHARS = Number.parseInt(process.env.REFINE_CONTEXT_LIMIT || '400000', 10) || 400000;
const STATUS_FILE = 'refine_status.json';

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

async function refineFile(filePath, relPath, refinedRoot, internalIndex, pathIndex, goldIndex, status, persistStatus) {
    const code = fs.readFileSync(filePath, 'utf8');

    // Skip very large files or vendor files if needed, but for now let's try all
    if (code.length > MAX_REFINE_CHARS) {
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

    const prompt = `
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

IMPORT RESOLUTION CONTEXT:
- Known internal module paths (same folder): ${localPaths.length > 0 ? `\n${localPaths.map(p => `  - ${p}`).join('\n')}` : 'None'}
- Known internal module paths (other folders, sample): ${otherPaths.length > 0 ? `\n${otherPaths.slice(0, 120).map(p => `  - ${p}`).join('\n')}` : 'None'}
- Known external packages: ${KNOWN_PACKAGES.length > 0 ? `\n${KNOWN_PACKAGES.map(p => `  - ${p}`).join('\n')}` : 'None'}
- Import resolution hints for this file: ${importHints.length > 0 ? `\n${importHints.slice(0, 60).map(h => `  - ${h.spec} -> ${h.suggestion} (${h.reason})`).join('\n')}` : 'None'}
- Unrecognized bare imports (do not invent packages): ${unresolvedBares.length > 0 ? `\n${unresolvedBares.slice(0, 60).map(s => `  - ${s}`).join('\n')}` : 'None'}

INSTRUCTIONS:
1. Restore clean control flow (use if/else instead of complex ternaries where appropriate).
2. Group related functions/variables logically.
3. Remove any remaining obfuscation artifacts (like proxy functions or unused helper calls).
4. Ensure the exports match a modern ESM/CommonJS structure.
5. Fix import specifiers so they resolve:
   - If an import matches a known internal module, rewrite it to the correct relative path.
   - If an import is a bare specifier, only use packages from the Known external packages list.
   - Do not invent new package names. If unsure, keep the original specifier.
6. Add helpful comments explaining complex logic blocks.
7. FIX any obviously broken logic caused by the assembly process (e.g. out-of-order definitions if detected).

FILE PATH: ${relPathPosix}

CODE:
\`\`\`javascript
${code}
\`\`\`

RESPONSE:
Output ONLY the refined JavaScript code. Do not include markdown blocks or preamble.
`;

    console.log(`[*] Refining ${relPath}...`);
    try {
        const refinedCode = await callLLM(prompt);
        // Remove markdown wrappers if the LLM ignored the instruction
        let cleaned = refinedCode.trim();
        if (cleaned.startsWith('```javascript')) {
            cleaned = cleaned.replace(/^```javascript\n/, '').replace(/\n```$/, '');
        } else if (cleaned.startsWith('```')) {
            cleaned = cleaned.replace(/^```\n/, '').replace(/\n```$/, '');
        }

        const refinedPath = path.join(refinedRoot, relPath);
        const refinedDir = path.dirname(refinedPath);
        if (!fs.existsSync(refinedDir)) fs.mkdirSync(refinedDir, { recursive: true });

        fs.writeFileSync(refinedPath, cleaned);
        console.log(`[+] Saved refined version: ${refinedPath}`);
        updateStatus(status, relPath, {
            status: 'done',
            sizeChars: code.length,
            refinedPath: toPosixPath(path.relative(OUTPUT_ROOT, refinedPath)),
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
    const latestBootstrapDir = getLatestBootstrapDir(BOOTSTRAP_ROOT);
    const goldIndex = latestBootstrapDir ? buildGoldIndex(latestBootstrapDir) : null;
    if (latestBootstrapDir) {
        console.log(`[*] Using latest bootstrap gold: ${toPosixPath(path.relative(OUTPUT_ROOT, latestBootstrapDir))}`);
    } else {
        console.warn(`[!] No bootstrap gold directory found at ${BOOTSTRAP_ROOT}`);
    }
    const limit = pLimit(PROVIDER === 'gemini' ? 2 : 2); // Conservative limit for all providers to avoid timeouts
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
        return limit(() => refineFile(file, relPath, refinedRoot, internalIndex, pathIndex, goldIndex, status, persistStatus));
    }).filter(Boolean);

    await Promise.all(tasks);
    await persistStatus();
    console.log(`[*] Refinement complete.`);
}

run().catch(console.error);
