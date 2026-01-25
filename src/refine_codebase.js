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

async function refineFile(filePath, relPath, refinedRoot, internalIndex, pathIndex, goldIndex, graphChunks, logicByName, logicRegistry, status, persistStatus) {
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
   - Do not invent new package names. If unsure, keep the original specifier.
6. Vendor libraries (e.g. zod/react) were filtered earlier. Do not create internal files or import paths named after vendor libraries (e.g. ./zod, ./react, src/zod/*).
7. Add helpful comments explaining complex logic blocks.
8. FIX any obviously broken logic caused by the assembly process (e.g. out-of-order definitions if detected).

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
        return limit(() => refineFile(file, relPath, refinedRoot, internalIndex, pathIndex, goldIndex, graphChunks, logicByName, logicRegistry, status, persistStatus));
    }).filter(Boolean);

    await Promise.all(tasks);
    await persistStatus();
    console.log(`[*] Refinement complete.`);
}

run().catch(console.error);
