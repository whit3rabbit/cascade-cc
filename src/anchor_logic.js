require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * PROTECTED_GLOBALS: Global variables that should never be renamed.
 * These are standard JavaScript/Node.js built-ins.
 */
const PROTECTED_GLOBALS = new Set([
    'Object', 'Array', 'String', 'Number', 'Boolean', 'Promise', 'Error', 'JSON', 'Math',
    'RegExp', 'Map', 'Set', 'WeakMap', 'WeakSet', 'globalThis', 'window', 'global',
    'process', 'require', 'module', 'exports', 'URL', 'Buffer', 'console', 'TypeError',
    'RangeError', 'ReferenceError', 'SyntaxError', 'URIError', 'EvalError', 'InternalError',
    'Intl', 'WebAssembly', 'Atomics', 'SharedArrayBuffer', 'DataView', 'ArrayBuffer',
    'Float32Array', 'Float64Array', 'Int8Array', 'Int16Array', 'Int32Array', 'Uint8Array',
    'Uint16Array', 'Uint32Array', 'Uint8ClampedArray', 'BigInt64Array', 'BigUint64Array',
    'arguments', 'undefined', 'null', 'true', 'false', 'Infinity', 'NaN', 'parseInt',
    'parseFloat', 'isNaN', 'isFinite', 'decodeURI', 'decodeURIComponent', 'encodeURI',
]);

/**
 * RESERVED_PROPERTIES: Property names that reflect standard API methods.
 * Renaming these would break standard library usage.
 */
const RESERVED_PROPERTIES = new Set([
    'then', 'catch', 'finally', 'length', 'map', 'forEach', 'filter', 'reduce',
    'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'join', 'split',
    'includes', 'indexOf', 'lastIndexOf', 'hasOwnProperty', 'toString',
    'valueOf', 'prototype', 'constructor', 'apply', 'call', 'bind',
    'message', 'stack', 'name', 'code', 'status', 'headers', 'body',
    'write', 'end', 'on', 'once', 'emit', 'removeListener', 'removeAllListeners',
    'substring', 'substr', 'replace', 'trim', 'toLowerCase', 'toUpperCase', 'charAt',
    'match', 'search', 'slice', 'concat', 'entries', 'keys', 'values', 'from',
    'stdout', 'stderr', 'stdin', 'destroyed', 'preInit'
]);

/**
 * Calculates cosine similarity (dot product) between two logic vectors.
 * Higher similarity indicates more similar structural logic.
 */
function calculateSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    return vecA.reduce((sum, a, i) => sum + a * vecB[i], 0); // Dot product
}

/**
 * Aligns symbols between two matched entities and updates the target mapping.
 * Uses structural key-based alignment for robustness against code reordering.
 * 
 * Workflow:
 * 1. Build maps of reference symbols for fast lookup.
 * 2. Attempt high-confidence "key-based" alignment (structural matching).
 * 3. Fallback to name-based alignment if necessary.
 * 4. Apply matches to targetMapping, filtering out protected/reserved names.
 */
function alignSymbols(targetMapping, resolvedVariables, resolvedProperties, targetSymbols, refSymbols, sourceLabel, options = {}) {
    let alignedCount = 0;
    let { lockConfidence = null } = options;

    // ADDITION: If a match comes from 'bootstrap' or label indicates golden match, lock it with 1.0 confidence
    if (sourceLabel && (sourceLabel.includes('bootstrap') || sourceLabel.includes('GOLDEN'))) {
        lockConfidence = 1.0;
    }

    // Create a map of ref symbols by their structural key and by name
    const refSymbolMap = new Map();
    const refNameMap = new Map();
    refSymbols.forEach(ref => {
        if (ref && typeof ref === 'object') {
            if (ref.key) refSymbolMap.set(ref.key, ref.name);
            refNameMap.set(ref.name, ref.name);
        }
    });

    // Strategy 1: Key-based alignment (Structural)
    // We match identifiers that have the same structural usage pattern (key).
    const matchesFound = [];
    for (const targetSymbol of targetSymbols) {
        if (!targetSymbol || typeof targetSymbol !== 'object' || !targetSymbol.key) continue;
        const refMangled = refSymbolMap.get(targetSymbol.key);
        if (refMangled) {
            matchesFound.push({ target: targetSymbol.name, ref: refMangled, method: 'key' });
        }
    }

    // Strategy 2: Fallback to Name-based alignment if Key-based yielded very low results
    if (matchesFound.length === 0) {
        for (const targetSymbol of targetSymbols) {
            if (!targetSymbol || typeof targetSymbol !== 'object') continue;
            if (refNameMap.has(targetSymbol.name)) {
                matchesFound.push({ target: targetSymbol.name, ref: targetSymbol.name, method: 'name' });
            }
        }
    }

    // Apply matches: Resolve mangled names back to their logical origins
    for (const match of matchesFound) {
        const targetMangled = match.target;
        const refMangled = match.ref;

        // Never rename standard JS/Node objects or common property names
        if (PROTECTED_GLOBALS.has(targetMangled) || RESERVED_PROPERTIES.has(targetMangled)) continue;

        const resolvedVar = resolvedVariables[refMangled];
        const resolvedProp = resolvedProperties[refMangled];

        // Filter: Avoid global collisions for 1-character variables (e.g., 'A', 'Q').
        // Properties are allowed since they are context-specific.
        if (targetMangled.length === 1 && !resolvedProp) continue;

        if (resolvedVar) {
            if (!targetMapping.variables[targetMangled]) {
                targetMapping.variables[targetMangled] = {
                    name: typeof resolvedVar === 'string' ? resolvedVar : resolvedVar.name,
                    confidence: lockConfidence ?? (match.method === 'key'
                        ? (parseFloat(process.env.ANCHOR_KEY_CONFIDENCE) || 0.9)
                        : (parseFloat(process.env.ANCHOR_NAME_CONFIDENCE) || 0.85)),
                    source: `anchored_${match.method}_${sourceLabel}`
                };
                alignedCount++;
            }
        }
        if (resolvedProp) {
            if (!targetMapping.properties[targetMangled]) {
                targetMapping.properties[targetMangled] = {
                    name: typeof resolvedProp === 'string' ? resolvedProp : resolvedProp.name,
                    confidence: lockConfidence ?? (match.method === 'key'
                        ? (parseFloat(process.env.ANCHOR_KEY_CONFIDENCE) || 0.9)
                        : (parseFloat(process.env.ANCHOR_NAME_CONFIDENCE) || 0.85)),
                    source: `anchored_${match.method}_${sourceLabel}`
                };
                alignedCount++;
            }
        }
    }

    return alignedCount;
}

/**
 * Main anchoring entry point.
 * Orchestrates:
 * 1. Vectorization of the target version using Python ML scripts.
 * 2. Comparison of target chunks against a Registry (known libraries) or a Reference Version.
 * 3. Symbol alignment and mapping generation.
 * 4. Updating graph_map.json with logical filename suggestions.
 */
async function anchorLogic(targetVersion, referenceVersion = null, baseDir = './cascade_graph_analysis') {
    const targetPath = path.resolve(baseDir, targetVersion);
    const pythonEnv = process.env.PYTHON_BIN || (fs.existsSync(path.join(__dirname, '../.venv/bin/python3'))
        ? path.join(__dirname, '../.venv/bin/python3')
        : fs.existsSync(path.join(__dirname, '../ml/venv/bin/python3'))
            ? path.join(__dirname, '../ml/venv/bin/python3')
            : 'python3');

    console.log(`[*] Vectorizing target version: ${targetVersion}`);
    try {
        execSync(`${pythonEnv} ml/vectorize.py ${targetPath}`, { stdio: 'inherit' });
    } catch (error) {
        throw new Error(`Python Vectorization failed: ${error.message}`);
    }

    if (!referenceVersion) {
        // Mode: Registry-based anchoring
        const registryPath = path.join(baseDir, 'logic_registry.json');
        if (!fs.existsSync(registryPath)) {
            console.log(`[!] No reference version provided and logic_registry.json not found. Stopping.`);
            return;
        }

        console.log(`[*] Performing Registry-based Anchoring...`);
        // Use a more memory-efficient loading strategy if possible, 
        // but for now, we rely on the 8GB heap.
        const registryData = fs.readFileSync(registryPath, 'utf8');
        const registry = JSON.parse(registryData);
        // Explicitly null out the raw string to free memory
        // registryData = null; // Can't null a const, and garbage collection will handle it, but we can scope it.

        const targetLogicDbPath = path.join(targetPath, 'metadata', 'logic_db.json');
        const targetLogicDb = JSON.parse(fs.readFileSync(targetLogicDbPath, 'utf8'));

        const targetMappingPath = path.join(targetPath, 'metadata', 'mapping.json');
        let targetMapping = { version: "1.2", variables: {}, properties: {}, processed_chunks: [], metadata: { total_renamed: 0, last_updated: new Date().toISOString() } };
        if (fs.existsSync(targetMappingPath)) {
            targetMapping = JSON.parse(fs.readFileSync(targetMappingPath, 'utf8'));
        }

        let matchedCount = 0;
        let totalNamesAnchored = 0;
        let existingMatchedCount = 0;
        let totalSimilarity = 0;
        let highSimCount = 0;

        const lockThreshold = parseFloat(process.env.ANCHOR_LOCK_THRESHOLD) || 0.98;
        const lockConfidence = parseFloat(process.env.ANCHOR_LOCK_CONFIDENCE) || 0.99;

        const registryEntries = Object.entries(registry);
        const simThreshold = parseFloat(process.env.ANCHOR_SIMILARITY_THRESHOLD) || 0.9;

        for (const targetChunk of targetLogicDb) {
            let bestMatch = { ref: null, similarity: -1, label: null };
            const targetVec = targetChunk.vector;

            // Heuristic Signal Extraction
            const kbDescription = targetChunk.kb_info ? (targetChunk.kb_info.description || '').toLowerCase() : '';
            const kbPath = targetChunk.kb_info ? (targetChunk.kb_info.suggested_path || '').toLowerCase() : '';
            let boostLib = null;

            // Simple library name extraction from KB hints (e.g. "ink" from "inkScreenRenderer")
            if (kbPath.includes('ink') || kbDescription.includes('ink')) boostLib = 'ink';
            else if (kbPath.includes('react') || kbDescription.includes('react')) boostLib = 'react';
            else if (kbPath.includes('zod') || kbDescription.includes('zod')) boostLib = 'zod';
            else if (kbPath.includes('lodash') || kbDescription.includes('lodash')) boostLib = 'lodash';
            else if (kbPath.includes('axios') || kbDescription.includes('axios')) boostLib = 'axios';
            else if (kbPath.includes('chalk') || kbDescription.includes('chalk')) boostLib = 'chalk';
            else if (kbPath.includes('commander')) boostLib = 'commander';
            // Add more common libs as needed, or make dynamic

            for (let i = 0; i < registryEntries.length; i++) {
                const [label, refData] = registryEntries[i];
                let sim = calculateSimilarity(targetVec, refData.vector);

                // Apply Heuristic Boost
                // If the registry label matches the KB hint, we trust the NN match much more easily.
                if (boostLib && label.toLowerCase().includes(boostLib)) {
                    sim += 0.15; // significant boost
                }

                if (sim > bestMatch.similarity) {
                    bestMatch = { ref: refData, similarity: sim, label: label, originalSim: sim - (boostLib && label.toLowerCase().includes(boostLib) ? 0.15 : 0) };
                }
            }

            // Lowered global threshold for better coverage (0.85 default instead of 0.9)
            const effectiveThreshold = parseFloat(process.env.ANCHOR_SIMILARITY_THRESHOLD) || 0.85;

            if (bestMatch.similarity > effectiveThreshold) {
                const isNewChunk = !targetMapping.processed_chunks.includes(targetChunk.name);
                const logPrefix = isNewChunk ? '[ANCHOR/REGISTRY] NEW MATCH' : '[ANCHOR/REGISTRY] EXISTING';

                totalSimilarity += bestMatch.originalSim; // Use real sim for stats
                highSimCount++;

                const boostTag = (bestMatch.similarity > bestMatch.originalSim) ? `[BOOSTED]` : '';
                console.log(`    ${logPrefix}: ${targetChunk.name} -> ${bestMatch.label} (${(bestMatch.originalSim * 100).toFixed(2)}% ${boostTag} => ${(bestMatch.similarity * 100).toFixed(2)}%)`);

                const namesAdded = alignSymbols(
                    targetMapping,
                    bestMatch.ref.resolved_variables,
                    bestMatch.ref.resolved_properties,
                    targetChunk.symbols,
                    bestMatch.ref.symbols,
                    bestMatch.label,
                    bestMatch.similarity >= lockThreshold ? { lockConfidence } : {}
                );

                if (bestMatch.similarity > 0.95) {
                    // This is a library match!
                    const isLibrary = bestMatch.label.includes('_') || bestMatch.label.includes('-v');
                    const libName = isLibrary ? bestMatch.label.split('_')[0] : 'unknown_lib';

                    targetChunk.category = 'vendor';
                    targetChunk.label = 'GOLDEN_LIBRARY_MATCH';
                    targetChunk.role = `VENDOR: ${libName}`;
                    targetChunk.isGoldenMatch = true;
                    // Propose a file path based on the library structure
                    targetChunk.proposedPath = `src/vendor/${libName}/${targetChunk.name}.ts`;
                }

                if (isNewChunk) {
                    targetMapping.processed_chunks.push(targetChunk.name);
                    matchedCount++;
                    totalNamesAnchored += namesAdded;
                } else {
                    existingMatchedCount++;
                }
            } else if (bestMatch.similarity > 0.65) {
                // Log near misses to help debug why things aren't matching
                const boostTag = (bestMatch.similarity > bestMatch.originalSim) ? `[BOOSTED]` : '';
                console.log(`    [ANCHOR/MISS] ${targetChunk.name} -> ${bestMatch.label} (${(bestMatch.originalSim * 100).toFixed(2)}% ${boostTag}) - Below threshold ${effectiveThreshold}`);
            }
        }
        fs.writeFileSync(targetMappingPath, JSON.stringify(targetMapping, null, 2));

        // Match anchoring results back to graph_map.json
        const graphMapPath = path.join(targetPath, 'metadata', 'graph_map.json');
        const graphMapRaw = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
        const chunks = Array.isArray(graphMapRaw) ? graphMapRaw : (graphMapRaw.chunks || []);

        for (const targetChunk of targetLogicDb) {
            const mapEntry = chunks.find(m => m.name === targetChunk.name);
            if (mapEntry) {
                if (targetChunk.suggestedFilename) mapEntry.suggestedFilename = targetChunk.suggestedFilename;
                if (targetChunk.proposedPath) mapEntry.proposedPath = targetChunk.proposedPath;
                if (targetChunk.isGoldenMatch) mapEntry.isGoldenMatch = targetChunk.isGoldenMatch;
                if (targetChunk.category) mapEntry.category = targetChunk.category;
                if (targetChunk.label) mapEntry.label = targetChunk.label;
                if (targetChunk.role) mapEntry.role = targetChunk.role;
            }
        }

        if (Array.isArray(graphMapRaw)) {
            fs.writeFileSync(graphMapPath, JSON.stringify(chunks, null, 2));
        } else {
            graphMapRaw.chunks = chunks;
            fs.writeFileSync(graphMapPath, JSON.stringify(graphMapRaw, null, 2));
        }
        const avgSim = highSimCount > 0 ? (totalSimilarity / highSimCount * 100).toFixed(2) : 0;

        console.log(`\n[+] Registry Anchoring complete.`);
        console.log(`    - Average Match Similarity: ${avgSim}%`);
        console.log(`    - Chunks Matched (Total):   ${highSimCount} / ${targetLogicDb.length}`);
        console.log(`    - Chunks Matched (New):     ${matchedCount}`);
        console.log(`    - Chunks Matched (Already): ${existingMatchedCount}`);
        console.log(`    - Total Aligned Symbols:    ${Object.keys(targetMapping.variables).length + Object.keys(targetMapping.properties).length}`);
        console.log(`    - New Symbols Added:        ${totalNamesAnchored}`);

    } else {
        // Mode: Version-to-version anchoring (Legacy/Direct)
        console.log(`[*] Performing Direct Anchoring: ${targetVersion} <-- ${referenceVersion}`);
        const referencePath = path.resolve(baseDir, referenceVersion);

        try {
            execSync(`${pythonEnv} ml/vectorize.py ${referencePath}`, { stdio: 'inherit' });
        } catch (error) {
            throw new Error(`Python Vectorization failed: ${error.message}`);
        }

        const targetLogicDb = JSON.parse(fs.readFileSync(path.join(targetPath, 'metadata', 'logic_db.json'), 'utf8'));
        const referenceLogicDb = JSON.parse(fs.readFileSync(path.join(referencePath, 'metadata', 'logic_db.json'), 'utf8'));
        const referenceMappingPath = path.join(referencePath, 'metadata', 'mapping.json');

        if (!fs.existsSync(referenceMappingPath)) {
            console.error(`[!] Reference mapping.json not found at ${referenceMappingPath}`);
            return;
        }
        const referenceMapping = JSON.parse(fs.readFileSync(referenceMappingPath, 'utf8'));

        const targetMappingPath = path.join(targetPath, 'metadata', 'mapping.json');
        let targetMapping = { version: "1.2", variables: {}, properties: {}, processed_chunks: [], metadata: { total_renamed: 0, last_updated: new Date().toISOString() } };
        if (fs.existsSync(targetMappingPath)) {
            targetMapping = JSON.parse(fs.readFileSync(targetMappingPath, 'utf8'));
        }

        let matchedCount = 0;
        let totalNamesAnchored = 0;
        let existingMatchedCount = 0;
        let totalSimilarity = 0;
        let highSimCount = 0;

        const lockThreshold = parseFloat(process.env.ANCHOR_LOCK_THRESHOLD) || 0.98;
        const lockConfidence = parseFloat(process.env.ANCHOR_LOCK_CONFIDENCE) || 0.99;

        for (const targetChunk of targetLogicDb) {
            let bestMatch = { ref: null, similarity: -1 };
            for (const refChunk of referenceLogicDb) {
                const sim = calculateSimilarity(targetChunk.vector, refChunk.vector);
                if (sim > bestMatch.similarity) {
                    bestMatch = { ref: refChunk, similarity: sim };
                }
            }

            if (bestMatch.similarity > (parseFloat(process.env.ANCHOR_SIMILARITY_THRESHOLD) || 0.9)) { // Lowered for cold start reliability
                totalSimilarity += bestMatch.similarity;
                highSimCount++;

                console.log(`    [ANCHOR/DIRECT] Match: ${targetChunk.name} -> ${bestMatch.ref.name} (${(bestMatch.similarity * 100).toFixed(2)}%)`);

                // For direct anchoring, we need to extract the relevant mappings from refMapping for THAT chunk
                const refVars = {};
                for (const [m, e] of Object.entries(referenceMapping.variables)) {
                    const match = Array.isArray(e) ? e.find(x => x.source === bestMatch.ref.name) : (e.source === bestMatch.ref.name ? e : null);
                    if (match) refVars[m] = match;
                }
                const refProps = {};
                for (const [m, e] of Object.entries(referenceMapping.properties)) {
                    const match = Array.isArray(e) ? e.find(x => x.source === bestMatch.ref.name) : (e.source === bestMatch.ref.name ? e : null);
                    if (match) refProps[m] = match;
                }

                const namesAdded = alignSymbols(
                    targetMapping,
                    refVars,
                    refProps,
                    targetChunk.symbols,
                    bestMatch.ref.symbols,
                    bestMatch.ref.name,
                    bestMatch.similarity >= lockThreshold ? { lockConfidence } : {}
                );
                if (!targetMapping.processed_chunks.includes(targetChunk.name)) {
                    targetMapping.processed_chunks.push(targetChunk.name);
                    matchedCount++;
                    totalNamesAnchored += namesAdded;
                } else {
                    existingMatchedCount++;
                }
            }
        }
        fs.writeFileSync(targetMappingPath, JSON.stringify(targetMapping, null, 2));
        const avgSim = highSimCount > 0 ? (totalSimilarity / highSimCount * 100).toFixed(2) : 0;

        console.log(`\n[+] Direct Anchoring complete.`);
        console.log(`    - Average Match Similarity: ${avgSim}%`);
        console.log(`    - Chunks Matched (Total):   ${highSimCount} / ${targetLogicDb.length}`);
        console.log(`    - Chunks Matched (New):     ${matchedCount}`);
        console.log(`    - Chunks Matched (Already): ${existingMatchedCount}`);
        console.log(`    - Total Aligned Symbols:    ${Object.keys(targetMapping.variables).length + Object.keys(targetMapping.properties).length}`);
        console.log(`    - New Symbols Added:        ${totalNamesAnchored}`);
    }
}

if (require.main === module) {
    const args = process.argv.slice(2);
    let targetVersion = args[0];

    if (!targetVersion) {
        // Auto-detect the latest version from cascade_graph_analysis
        const baseDir = './cascade_graph_analysis';
        const dirs = fs.readdirSync(baseDir).filter(d => {
            return fs.statSync(path.join(baseDir, d)).isDirectory() && d !== 'bootstrap';
        }).sort().reverse();

        if (dirs.length > 0) {
            targetVersion = dirs[0];
            console.log(`[*] No version specified. Auto-detected latest: ${targetVersion}`);
        } else {
            console.log("Usage: node src/anchor_logic.js <target_version> [reference_version]");
            process.exit(1);
        }
    }

    anchorLogic(targetVersion, args[1]).catch(console.error);
}

module.exports = { anchorLogic };
