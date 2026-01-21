require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const pLimit = require('p-limit');

const OUTPUT_ROOT = './cascade_graph_analysis';
const KB_PATH = './knowledge_base.json';
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

async function refineFile(filePath, relPath, internalIndex, pathIndex) {
    const code = fs.readFileSync(filePath, 'utf8');

    // Skip very large files or vendor files if needed, but for now let's try all
    if (code.length > 100000) {
        console.warn(`[!] Skipping ${relPath} (too large: ${code.length} chars)`);
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

    const prompt = `
Role: Senior Staff Software Engineer / Reverse Engineer
Task: Source Code Reconstruction & Logic Refinement

I have an assembled JavaScript file that was deobfuscated from a minified bundle.
The identifiers names are mostly correct, but the logic structure might still be "minified" (e.g., flattened loops, complex ternary chains, inlined constants, dead code branches).

GOAL: Reconstruct this file into what the ORIGINAL source code likely looked like.

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

        const refinedPath = filePath.replace('/assemble/', '/refined_assemble/');
        const refinedDir = path.dirname(refinedPath);
        if (!fs.existsSync(refinedDir)) fs.mkdirSync(refinedDir, { recursive: true });

        fs.writeFileSync(refinedPath, cleaned);
        console.log(`[+] Saved refined version: ${refinedPath}`);
    } catch (err) {
        console.error(`[!] Error refining ${relPath}: ${err.message}`);
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
    const limit = pLimit(PROVIDER === 'gemini' ? 2 : 2); // Conservative limit for all providers to avoid timeouts
    const tasks = files.map(file => limit(() => refineFile(file, path.relative(assembleDir, file), internalIndex, pathIndex)));

    await Promise.all(tasks);
    console.log(`[*] Refinement complete.`);
}

run().catch(console.error);
