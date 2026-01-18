const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const VERSION = '2.1.12';
const SYSTEM_MAP_PATH = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'system_map.v2.json');
const GRAPH_MAP_PATH = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'metadata', 'graph_map.json');
const CHUNKS_DIR = path.join(__dirname, '..', 'cascade_graph_analysis', VERSION, 'chunks');

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
    Object.values(systemMap.modules).forEach(mod => {
        mod.original_chunks.forEach(chunk => {
            analyzedChunks.add(chunk.split('_')[0].replace('.js', ''));
        });
    });

    console.log(`[*] Currently mapped chunks: ${analyzedChunks.size}`);

    const frontier = [];
    const visited = new Set(analyzedChunks);

    // Identify unanalyzed neighbors
    graphMap.chunks.forEach(chunk => {
        const chunkId = chunk.name.split('_')[0];
        if (analyzedChunks.has(chunkId)) {
            // Find its outbound neighbors that aren't analyzed
            (chunk.outbound || []).forEach(neighbor => {
                const neighborId = neighbor.split('_')[0];
                if (!visited.has(neighborId)) {
                    frontier.push({
                        id: neighborId,
                        fullName: neighbor,
                        centrality: chunk.centrality || 0, // Simplified: use source's centrality as a proxy if neighbor isn't in top list
                        inDegree: chunk.inDegree || 0
                    });
                    visited.add(neighborId);
                }
            });
        }
    });

    // Also include high-centrality chunks that aren't mapped yet (global seeds)
    graphMap.chunks.forEach(chunk => {
        const chunkId = chunk.name.split('_')[0];
        if (!visited.has(chunkId) && (chunk.centrality > 0.1 || (chunk.outbound && chunk.outbound.length > 5))) {
            frontier.push({
                id: chunkId,
                fullName: chunk.name,
                centrality: chunk.centrality || 0,
                inDegree: chunk.inDegree || 0
            });
            visited.add(chunkId);
        }
    });

    // Sort frontier by "importance" (centrality and outbound connections)
    return frontier.sort((a, b) => b.centrality - a.centrality || b.inDegree - a.inDegree);
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

const args = process.argv.slice(2);
const batchSize = parseInt(args[0]) || 10;
const dryRun = args.includes('--dry-run');

runCrawler(batchSize, dryRun);
