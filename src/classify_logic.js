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
    const familySet = new Set();
    graphData.chunks.forEach(node => {
        const isFounder = node.hasTengu || node.hasGenerator || node.hasStateMutator || !!node.error_signature;
        if (isFounder) {
            familySet.add(node.name);
            node.category = 'founder';
        }
    });
    console.log(`    [+] Identified ${familySet.size} founder seed chunks.`);

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

    // 4. Role Assignment
    const vendorIn = parseInt(process.env.VENDOR_LIBRARY_IN_DEGREE) || 15;
    const vendorOut = parseInt(process.env.VENDOR_LIBRARY_OUT_DEGREE) || 5;
    const coreIn = parseInt(process.env.CORE_ORCHESTRATOR_IN_DEGREE) || 5;
    const coreOut = parseInt(process.env.CORE_ORCHESTRATOR_OUT_DEGREE) || 5;

    graphData.chunks.forEach(node => {
        const inCount = inDegree.get(node.name) || 0;
        const outCount = (node.outbound || []).length;
        const isFamily = familySet.has(node.name);
        const isAnchored = nnMapping.processed_chunks.includes(node.name);

        if (isAnchored) {
            node.category = 'vendor';
            node.label = 'LIBRARY_MATCH';
            const matchEntry = varSourceMap.get(node.name);
            if (matchEntry) node.role = `LIB: ${matchEntry.name}`;
            node.isGoldenMatch = true;
        } else if (isFamily && node.category !== 'founder') {
            node.category = 'family';
        } else if (!isFamily) {
            node.category = 'vendor';
        }

        if (!isFamily && inCount > vendorIn && outCount < vendorOut) {
            node.label = 'VENDOR_LIBRARY';
        } else if (isFamily && inCount > coreIn && outCount > coreOut) {
            node.label = 'CORE_ORCHESTRATOR';
        }

        // Apply fallback roles
        if (!node.role || node.role === 'MODULE') {
            if (node.hasGenerator) node.role = 'STREAM_ORCHESTRATOR';
            // Entry signals
            if (node.entrySignalCount > 0) node.role = 'CLI_MODULE';
        }
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
