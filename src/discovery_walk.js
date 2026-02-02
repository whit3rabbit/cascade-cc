const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const DEFAULT_VERSION = '2.1.12';

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const filteredArgs = args.filter(arg => arg !== '--dry-run');
const versionArg = filteredArgs[0];
const batchSizeArg = filteredArgs[1];

const VERSION = versionArg || process.env.DISCOVERY_VERSION || DEFAULT_VERSION;
const SYSTEM_MAP_PATH = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'system_map.v2.json');
const GRAPH_MAP_PATH = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'metadata', 'graph_map.json');
const CHUNKS_DIR = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'chunks');
const NEIGHBOR_BOOSTS_PATH = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'metadata', 'neighbor_boosts.json');

const LIB_HINTS = ['crypto', 'axios', 'lodash', 'zod', 'react', 'ink', 'chalk', 'commander'];

function extractLibHint(input) {
    if (!input || typeof input !== 'string') return null;
    const lower = input.toLowerCase();
    return LIB_HINTS.find(h => lower.includes(h)) || null;
}

function loadJSON(filePath) {
    if (!fs.existsSync(filePath)) return null;
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (e) {
        console.error(`[!] Failed to parse JSON: ${filePath}`);
        return null;
    }
}

function getDiscoveryFrontier() {
    const systemMap = loadJSON(SYSTEM_MAP_PATH);
    const graphMap = loadJSON(GRAPH_MAP_PATH);

    if (!systemMap || !graphMap) {
        console.error('[!] Missing system map or graph map.');
        return [];
    }

    // Get all analyzed chunks (normalized to basename)
    const analyzedChunks = new Set();
    const modules = systemMap.modules || {};
    Object.values(modules).forEach(mod => {
        (mod.original_chunks || []).forEach(chunk => {
            analyzedChunks.add(chunk.split('_')[0].replace('.js', ''));
        });
    });

    console.log(`[*] Currently mapped chunks: ${analyzedChunks.size}`);

    const frontier = [];
    const visited = new Set(analyzedChunks);

    const graphChunks = Array.isArray(graphMap) ? graphMap : (graphMap.chunks || []);
    const chunkByBaseId = new Map();
    graphChunks.forEach(chunk => {
        const baseId = chunk.name.split('_')[0];
        if (!chunkByBaseId.has(baseId)) chunkByBaseId.set(baseId, chunk);
    });

    const neighborBoosts = new Map();
    Object.values(modules).forEach(mod => {
        const confidence = mod.confidence || mod.similarity || mod.matchSimilarity || 0;
        if (confidence < 0.95) return;
        const libHint = extractLibHint(mod.path || mod.proposedPath || mod.name || mod.displayName || '');
        if (!libHint) return;
        (mod.original_chunks || []).forEach(chunkName => {
            const baseId = chunkName.split('_')[0].replace('.js', '');
            const graphChunk = chunkByBaseId.get(baseId);
            if (!graphChunk) return;
            (graphChunk.outbound || []).forEach(neighborName => {
                const neighborId = neighborName.split('_')[0];
                const prev = neighborBoosts.get(neighborId);
                if (!prev || prev.confidence < confidence) {
                    neighborBoosts.set(neighborId, { lib: libHint, confidence, source: baseId });
                }
            });
        });
    });

    // Identify unanalyzed neighbors
    graphChunks.forEach(chunk => {
        const chunkId = chunk.name.split('_')[0];
        if (analyzedChunks.has(chunkId)) {
            // Find its outbound neighbors that aren't analyzed
            (chunk.outbound || []).forEach(neighbor => {
                const neighborId = neighbor.split('_')[0];
                if (!visited.has(neighborId)) {
                    const boost = neighborBoosts.get(neighborId);
                    frontier.push({
                        id: neighborId,
                        fullName: neighbor,
                        centrality: chunk.centrality || 0, // Simplified: use source's centrality as a proxy if neighbor isn't in top list
                        inDegree: chunk.inDegree || 0,
                        boostLib: boost ? boost.lib : null,
                        boostConfidence: boost ? boost.confidence : 0
                    });
                    visited.add(neighborId);
                }
            });
        }
    });

    // Also include high-centrality chunks that aren't mapped yet (global seeds)
    graphChunks.forEach(chunk => {
        const chunkId = chunk.name.split('_')[0];
        if (!visited.has(chunkId) && (chunk.centrality > 0.1 || (chunk.outbound && chunk.outbound.length > 5))) {
            const boost = neighborBoosts.get(chunkId);
            frontier.push({
                id: chunkId,
                fullName: chunk.name,
                centrality: chunk.centrality || 0,
                inDegree: chunk.inDegree || 0,
                boostLib: boost ? boost.lib : null,
                boostConfidence: boost ? boost.confidence : 0
            });
            visited.add(chunkId);
        }
    });

    if (neighborBoosts.size > 0) {
        fs.mkdirSync(path.dirname(NEIGHBOR_BOOSTS_PATH), { recursive: true });
        fs.writeFileSync(NEIGHBOR_BOOSTS_PATH, JSON.stringify(Object.fromEntries(neighborBoosts), null, 2));
    }

    // Sort frontier by "importance" (boosts first, then centrality and outbound connections)
    return frontier.sort((a, b) => (b.boostConfidence || 0) - (a.boostConfidence || 0) || b.centrality - a.centrality || b.inDegree - a.inDegree);
}

function runCrawler(batchSize = 20, dryRun = false) {
    const frontier = getDiscoveryFrontier();
    console.log(`[*] Discovery Frontier Size: ${frontier.length}`);

    const batch = frontier.slice(0, batchSize);
    console.log(`[*] Starting batch analysis for ${batch.length} chunks...`);

    // Build a lookup map for actual files in the chunks directory
    const filesInDir = fs.readdirSync(CHUNKS_DIR);
    const chunkFileMap = {};
    filesInDir.forEach(file => {
        if (file.endsWith('.js')) {
            const id = file.split('_')[0].replace('.js', '');
            chunkFileMap[id] = file;
        }
    });

    batch.forEach((target, index) => {
        // Resolve the actual filename (e.g., chunk214 -> chunk214_fileUtils.js)
        const chunkFile = chunkFileMap[target.id] || (target.fullName.endsWith('.js') ? target.fullName : `${target.fullName}.js`);

        if (!fs.existsSync(path.join(CHUNKS_DIR, chunkFile))) {
            console.log(`[${index + 1}/${batch.length}] [!] Chunk file not found: ${chunkFile}`);
            return;
        }

        console.log(`[${index + 1}/${batch.length}] Analyzing ${chunkFile} (Centrality: ${target.centrality.toFixed(4)})`);

        if (dryRun) {
            console.log(`    [DRY RUN] node src/architect_analyze.js ${chunkFile}`);
        } else {
            try {
                execSync(`node src/architect_analyze.js ${chunkFile}`, { stdio: 'inherit', cwd: path.join(__dirname, '..') });
            } catch (e) {
                console.error(`[!] Failed to analyze ${chunkFile}: ${e.message}`);
            }
        }
    });

    console.log('[*] Batch analysis complete.');
}

const batchSize = parseInt(batchSizeArg, 10) || 10;

runCrawler(batchSize, dryRun);
