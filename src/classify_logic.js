require('dotenv').config();
const fs = require('fs');
const path = require('path');

/**
 * ARCHITECTURAL CLASSIFIER (Phase 4 & 5)
 * This script runs after Static Analysis and Neural Anchoring.
 * It uses the enriched metadata to assign roles and identify the entry point.
 */
async function classifyLogic(targetVersion, baseDir = './cascade_graph_analysis') {
    const targetPath = path.resolve(baseDir, targetVersion);
    const graphDataPath = path.join(targetPath, 'metadata', 'graph_map.json');
    const mappingPath = path.join(targetPath, 'metadata', 'mapping.json');

    if (!fs.existsSync(graphDataPath)) {
        throw new Error(`Graph data not found at ${graphDataPath}`);
    }

    console.log(`[*] Phase 4 & 5: Architectural Classification (Version: ${targetVersion})...`);
    const graphData = JSON.parse(fs.readFileSync(graphDataPath, 'utf8'));
    const nnMapping = fs.existsSync(mappingPath) ? JSON.parse(fs.readFileSync(mappingPath, 'utf8')) : { processed_chunks: [], variables: {} };

    // Optimize variable lookup (same as in analyze.js previously)
    const varSourceMap = new Map();
    if (nnMapping.variables) {
        for (const entry of Object.values(nnMapping.variables)) {
            if (entry.source) varSourceMap.set(entry.source, entry);
        }
    }

    // 1. Pre-calculate Metrics (In-Degree)
    const inDegree = new Map();
    const nodesMap = new Map();
    graphData.chunks.forEach(node => {
        nodesMap.set(node.name, node);
        (node.outbound || []).forEach(neighbor => {
            inDegree.set(neighbor, (inDegree.get(neighbor) || 0) + 1);
        });
    });

    // 2. Identify "Founder Seeds" (Flags from Static Analysis)
    // 2. Identify Seeds (Strategic Shift: Inversion of Proof)
    const familySet = new Set();

    graphData.chunks.forEach(node => {
        const matchMeta = nnMapping.matches ? nnMapping.matches[node.name] : null;
        const similarity = matchMeta ? matchMeta.similarity : 0;

        // EVIDENCE FOR VENDOR: Only if NN is very sure (> 0.92)
        const isProvenLibrary = similarity > 0.92 || node.isGoldenMatch;

        // EVIDENCE FOR FOUNDER: 
        // a) Explicit signals (Tengu, Generators, Entry Points)
        const hasHardSignal = node.hasTengu || node.hasGenerator || node.entrySignalCount > 0;
        // b) Architectural importance WITHOUT library identity
        const isImportantOrphan = node.centrality > 0.01 && !isProvenLibrary;

        if (hasHardSignal || isImportantOrphan) {
            familySet.add(node.name);
            node.category = 'founder';
        } else if (isProvenLibrary) {
            node.category = 'vendor';
        } else {
            // THE TIPPING POINT: If we can't prove it's a library, 
            // and it's not a leaf-node utility, it's probably proprietary.
            node.category = 'family';
            familySet.add(node.name);
        }
    });
    console.log(`    [+] Identified ${familySet.size} founder/family chunks.`);

    // 3. Spreading Activation
    let changed = true;
    let iteration = 0;
    const spreadingRatio = parseFloat(process.env.SPREADING_THRESHOLD_RATIO) || 0.3;
    const spreadingCount = parseInt(process.env.SPREADING_THRESHOLD_COUNT) || 2;

    while (changed && iteration < 10) {
        iteration++;
        let startSize = familySet.size;
        graphData.chunks.forEach(node => {
            if (familySet.has(node.name)) return;

            const neighbors = node.outbound || [];
            const familyNeighbors = neighbors.filter(n => familySet.has(n));

            if (neighbors.length > 0 && (familyNeighbors.length / neighbors.length >= spreadingRatio || familyNeighbors.length >= spreadingCount)) {
                familySet.add(node.name);
            }
        });
        changed = familySet.size > startSize;
    }
    console.log(`    [+] Spreading complete: ${familySet.size} chunks in family set.`);

    // 4. Role Assignment (The Brain)
    graphData.chunks.forEach(node => {
        const matchMeta = nnMapping.matches ? nnMapping.matches[node.name] : null;
        const isLibrary = node.category === 'vendor';
        const isFounder = node.category === 'founder' || node.category === 'family';

        // --- 1. Known Vendor Roles (Auto-populated from Registry) ---
        if (isLibrary && matchMeta?.label) {
            const libBase = matchMeta.label.split('_')[0]; // Gets "zod", "react", etc.
            node.role = `LIB: ${libBase.toUpperCase()}`;
            node.label = 'VENDOR_LIBRARY';
            node.isGoldenMatch = true;
        }

        // --- 2. Founder Roles (Semantic Inference) ---
        else if (isFounder) {
            // Assign roles based on capabilities found during static analysis
            if (node.entrySignalCount > 0) {
                node.role = 'CLI_ENTRY_POINT';
            } else if (node.hasGenerator) {
                node.role = 'STREAM_PROCESSOR';
            } else if (node.hasNetwork) {
                node.role = 'API_CLIENT';
            } else if (node.hasFS) {
                node.role = 'FILESYSTEM_SERVICE';
            } else if (node.hasStateMutator) {
                node.role = 'STATE_ORCHESTRATOR';
            } else {
                node.role = 'BUSINESS_LOGIC'; // Default for Founder
            }

            node.label = (node.centrality > 0.05) ? 'CORE_MODULE' : 'INTERNAL_HELPER';
        }

        // --- 3. Default Fallback ---
        if (!node.role) node.role = 'UNKNOWN_MODULE';
    });

    // 5. Entry Point Identification
    console.log(`[*] Phase 5: Identifying Main Entry Point...`);
    let bestEntry = null;
    let maxEntryScore = -1;

    graphData.chunks.forEach(node => {
        const entryScore = (node.centrality * 10) + (node.entrySignalCount || 0);
        if (entryScore > maxEntryScore) {
            maxEntryScore = entryScore;
            bestEntry = node;
        }
    });

    if (bestEntry) {
        console.log(`    [+] Detected Entry Point: ${bestEntry.name} (Pretty Name: ${bestEntry.displayName || 'None'})`);
        bestEntry.role = 'ENTRY_POINT';
        bestEntry.category = 'founder';
    }

    // 6. Save Updated Metadata
    fs.writeFileSync(graphDataPath, JSON.stringify(graphData, null, 2));
    console.log(`[COMPLETE] Architectural Classification saved to ${graphDataPath}`);
}

if (require.main === module) {
    const args = process.argv.slice(2);
    let targetVersion = args[0];

    if (!targetVersion) {
        const baseDir = './cascade_graph_analysis';
        const dirs = fs.readdirSync(baseDir).filter(d => {
            return fs.statSync(path.join(baseDir, d)).isDirectory() && d !== 'bootstrap';
        }).sort().reverse();

        if (dirs.length > 0) {
            targetVersion = dirs[0];
            console.log(`[*] No version specified. Auto-detected latest: ${targetVersion}`);
        } else {
            console.log("Usage: node src/classify_logic.js <target_version>");
            process.exit(1);
        }
    }
    classifyLogic(targetVersion).catch(console.error);
}

module.exports = { classifyLogic };
