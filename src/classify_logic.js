require('dotenv').config();
const fs = require('fs');
const path = require('path');

/**
 * ARCHITECTURAL CLASSIFIER (Phase 4)
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

    console.log(`[*] Phase 4: Architectural Classification (Version: ${targetVersion})...`);
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

        // NEW: Custom Gold Detection
        // Checks if the match label indicates our custom proprietary gold standard
        const isCustomGold = matchMeta && matchMeta.label && matchMeta.label.includes('custom_claude_gold');

        // EVIDENCE FOR VENDOR: Only if NN is very sure (> 0.92)
        // PROTECT FIRST_PARTY: If name contains "claude" or "theme", OR if it matches custom gold, assume it's ours.
        const lowerName = (node.displayName || node.name || "").toLowerCase();
        const looksLikeFirstParty = lowerName.includes('claude') || lowerName.includes('theme') || isCustomGold;

        const isProvenLibrary = (similarity > 0.92 || node.isGoldenMatch) && !looksLikeFirstParty;

        // EVIDENCE FOR FOUNDER: 
        // a) Explicit signals (Tengu, Generators, Entry Points)
        const hasHardSignal = node.hasTengu || node.hasGenerator || node.entrySignalCount > 0;
        // b) Architectural importance WITHOUT library identity
        const isImportantOrphan = node.centrality > 0.01 && !isProvenLibrary;

        if (isCustomGold) {
            node.category = 'founder';
            node.isGoldenMatch = true;
            node.label = 'CONFIRMED_PROPRIETARY';

            // Auto-assign the path you provided in your custom_gold folder
            if (matchMeta.ref && matchMeta.ref.proposedPath) {
                node.proposedPath = matchMeta.ref.proposedPath;
            }
            familySet.add(node.name);
        } else if (hasHardSignal || isImportantOrphan) {
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
            let libBase = matchMeta.label.split('_')[0]; // Gets "zod", "react", etc.
            // SANITIZATION: If libBase contains non-alphanumeric chars (e.g. "require(.."), use "vendor_misc"
            if (/[^a-zA-Z0-9-]/.test(libBase)) {
                libBase = 'vendor_misc';
            }
            node.role = `LIB: ${libBase.toUpperCase()}`;
            node.label = 'VENDOR_LIBRARY';
            node.isGoldenMatch = true;
        }

        // --- 2. Founder Roles (Semantic Inference) ---
        else if (isFounder) {
            // Assign roles based on capabilities found during static analysis
            const codeLower = node.code?.toLowerCase() || "";
            if (node.entrySignalCount > 0 || (codeLower.includes('command') && codeLower.includes('parse'))) {
                node.role = 'CLI_COMMAND';
            } else if (node.hasGenerator) {
                node.role = 'STREAM_PROCESSOR';
            } else if (codeLower.includes('auth') || codeLower.includes('login') || codeLower.includes('token')) {
                node.role = 'AUTH_SERVICE';
            } else if (codeLower.includes('react') || codeLower.includes('ink') || codeLower.includes('usecontext')) {
                node.role = 'UI_COMPONENT';
            } else if (node.hasNetwork) {
                node.role = 'API_CLIENT';
            } else if (node.hasFS) {
                node.role = 'FILESYSTEM_SERVICE';
            } else if (node.hasStateMutator || codeLower.includes('setstate')) {
                node.role = 'STATE_ORCHESTRATOR';
            } else if (codeLower.includes('validate') || codeLower.includes('schema') || codeLower.includes('zod')) {
                node.role = 'VALIDATION_LOGIC';
            } else {
                node.role = 'UTILITY_HELPER';
            }

            node.label = (node.centrality > 0.05) ? 'CORE_MODULE' : 'INTERNAL_HELPER';
        }

        // --- 3. Default Fallback & Sanitization ---
        if (!node.role || node.role.includes('require(') || node.role.includes('tmp/') || node.role.length > 50) {
            node.role = 'UNCLASSIFIED_LOGIC';
        }

        if (node.suggestedFilename === '-') node.suggestedFilename = null;
        if (node.proposedPath === '-') node.proposedPath = null;

        // --- 4. Role-based Path Hinting (New) ---
        const roleToFolderMap = {
            'STREAM_ORCHESTRATOR': 'src/core/session',
            'STREAM_PROCESSOR': 'src/services/streaming',
            'API_CLIENT': 'src/services/network',
            'FILESYSTEM_SERVICE': 'src/services/fs',
            'UI_COMPONENT': 'src/components',
            'CLI_COMMAND': 'src/commands',
            'STATE_ORCHESTRATOR': 'src/services/state',
            'AUTH_SERVICE': 'src/services/auth',
            'VALIDATION_LOGIC': 'src/utils/validation',
            'UTILITY_HELPER': 'src/utils'
        };

        if (node.kb_info && node.kb_info.suggested_path && node.kb_info.suggested_path !== '-') {
            node.proposedPath = node.kb_info.suggested_path.replace(/`/g, '');
        }

        if (roleToFolderMap[node.role] && !node.proposedPath) {
            node.proposedPath = `${roleToFolderMap[node.role]}/${node.displayName || node.suggestedFilename || node.name}.ts`;
        }
    });

    // 5. Entry Point Identification
    console.log(`[*] Phase 4.1: Identifying Main Entry Point...`);
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
    const versionIdx = args.indexOf('--version');
    let targetVersion = null;
    if (versionIdx !== -1) {
        targetVersion = args[versionIdx + 1];
    } else {
        for (const arg of args) {
            if (arg.startsWith('--')) continue;
            targetVersion = arg;
            break;
        }
    }

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
