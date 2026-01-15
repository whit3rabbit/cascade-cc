const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

function calculateSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    return vecA.reduce((sum, a, i) => sum + a * vecB[i], 0); // Dot product
}

/**
 * Aligns symbols between two matched entities and updates the target mapping.
 */
function alignSymbols(targetMapping, resolvedVariables, resolvedProperties, targetSymbols, refSymbols, sourceLabel) {
    let alignedCount = 0;
    const minLen = Math.min(targetSymbols.length, refSymbols.length);
    for (let i = 0; i < minLen; i++) {
        const targetMangled = targetSymbols[i];
        const refMangled = refSymbols[i];

        // Check registry or reference mapping for resolved name
        let resolved = resolvedVariables[refMangled] || resolvedProperties[refMangled];

        if (resolved) {
            // Handle array of possibilities if applicable (though usually it's one)
            const isProperty = !!resolvedProperties[refMangled];
            const targetMap = isProperty ? targetMapping.properties : targetMapping.variables;

            if (!targetMap[targetMangled]) {
                targetMap[targetMangled] = {
                    name: typeof resolved === 'string' ? resolved : resolved.name,
                    confidence: 0.95,
                    source: `anchored_from_${sourceLabel}`
                };
                alignedCount++;
            }
        }
    }
    return alignedCount;
}

async function anchorLogic(targetVersion, referenceVersion = null, baseDir = './cascade_graph_analysis') {
    const targetPath = path.resolve(baseDir, targetVersion);
    const pythonEnv = fs.existsSync(path.join(__dirname, '../.venv/bin/python3'))
        ? path.join(__dirname, '../.venv/bin/python3')
        : fs.existsSync(path.join(__dirname, '../ml/venv/bin/python3'))
            ? path.join(__dirname, '../ml/venv/bin/python3')
            : 'python3';

    console.log(`[*] Vectorizing target version: ${targetVersion}`);
    try {
        execSync(`${pythonEnv} ml/vectorize.py ${targetPath}`, { stdio: 'inherit' });
    } catch (error) {
        console.error(`[!] Error during vectorization: ${error.message}`);
        return;
    }

    if (!referenceVersion) {
        // Mode: Registry-based anchoring
        const registryPath = path.join(baseDir, 'logic_registry.json');
        if (!fs.existsSync(registryPath)) {
            console.log(`[!] No reference version provided and logic_registry.json not found. Stopping.`);
            return;
        }

        console.log(`[*] Performing Registry-based Anchoring...`);
        const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
        const targetLogicDb = JSON.parse(fs.readFileSync(path.join(targetPath, 'metadata', 'logic_db.json'), 'utf8'));

        const targetMappingPath = path.join(targetPath, 'metadata', 'mapping.json');
        let targetMapping = { version: "1.2", variables: {}, properties: {}, processed_chunks: [] };
        if (fs.existsSync(targetMappingPath)) {
            targetMapping = JSON.parse(fs.readFileSync(targetMappingPath, 'utf8'));
        }

        let matchedCount = 0;
        let totalNamesAnchored = 0;

        for (const targetChunk of targetLogicDb) {
            let bestMatch = { ref: null, similarity: -1, label: null };
            for (const [label, refData] of Object.entries(registry)) {
                const sim = calculateSimilarity(targetChunk.vector, refData.vector);
                if (sim > bestMatch.similarity) {
                    bestMatch = { ref: refData, similarity: sim, label: label };
                }
            }

            if (bestMatch.similarity > 0.98) {
                console.log(`    [ANCHOR/REGISTRY] Match: ${targetChunk.name} -> ${bestMatch.label} (${(bestMatch.similarity * 100).toFixed(2)}%)`);
                const namesAdded = alignSymbols(
                    targetMapping,
                    bestMatch.ref.resolved_variables,
                    bestMatch.ref.resolved_properties,
                    targetChunk.symbols,
                    bestMatch.ref.symbols,
                    bestMatch.label
                );
                if (!targetMapping.processed_chunks.includes(targetChunk.name)) {
                    targetMapping.processed_chunks.push(targetChunk.name);
                    matchedCount++;
                    totalNamesAnchored += namesAdded;
                }
            }
        }
        fs.writeFileSync(targetMappingPath, JSON.stringify(targetMapping, null, 2));
        console.log(`[+] Registry Anchoring complete. Matched ${matchedCount} chunks, aligned ${totalNamesAnchored} symbols.`);

    } else {
        // Mode: Version-to-version anchoring (Legacy/Direct)
        console.log(`[*] Performing Direct Anchoring: ${targetVersion} <-- ${referenceVersion}`);
        const referencePath = path.resolve(baseDir, referenceVersion);

        try {
            execSync(`${pythonEnv} ml/vectorize.py ${referencePath}`, { stdio: 'inherit' });
        } catch (error) {
            console.error(`[!] Error during reference vectorization: ${error.message}`);
            return;
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
        let targetMapping = { version: "1.2", variables: {}, properties: {}, processed_chunks: [] };
        if (fs.existsSync(targetMappingPath)) {
            targetMapping = JSON.parse(fs.readFileSync(targetMappingPath, 'utf8'));
        }

        let matchedCount = 0;
        let totalNamesAnchored = 0;

        for (const targetChunk of targetLogicDb) {
            let bestMatch = { ref: null, similarity: -1 };
            for (const refChunk of referenceLogicDb) {
                const sim = calculateSimilarity(targetChunk.vector, refChunk.vector);
                if (sim > bestMatch.similarity) {
                    bestMatch = { ref: refChunk, similarity: sim };
                }
            }

            if (bestMatch.similarity > 0.98) {
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
                    bestMatch.ref.name
                );
                if (!targetMapping.processed_chunks.includes(targetChunk.name)) {
                    targetMapping.processed_chunks.push(targetChunk.name);
                    matchedCount++;
                    totalNamesAnchored += namesAdded;
                }
            }
        }
        fs.writeFileSync(targetMappingPath, JSON.stringify(targetMapping, null, 2));
        console.log(`[+] Direct Anchoring complete. Matched ${matchedCount} chunks, aligned ${totalNamesAnchored} symbols.`);
    }
}

if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.log("Usage: node src/anchor_logic.js <target_version> [reference_version]");
        process.exit(1);
    }
    anchorLogic(args[0], args[1]).catch(console.error);
}

module.exports = { anchorLogic };
