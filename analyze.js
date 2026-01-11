const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const { encode } = require('gpt-tokenizer');

// --- 1. CONFIGURATION ---
const PACKAGE_NAME = '@anthropic-ai/claude-code';
const ANALYSIS_DIR = './claude-analysis';
const OUTPUT_BASE = './cascade_graph_analysis';
const SIGNAL_KEYWORDS = ['anthropic', 'claude', 'mcp', 'agent', 'terminal', 'prompt', 'session', 'protocol', 'codeloop'];

function ensureTargetExists() {
    // 1. Check command line argument
    if (process.argv[2]) {
        return process.argv[2];
    }

    // 2. Check current directory
    if (fs.existsSync('./cli.js')) {
        return './cli.js';
    }

    // 3. Search claude-analysis versioned folders
    if (fs.existsSync(ANALYSIS_DIR)) {
        const versions = fs.readdirSync(ANALYSIS_DIR);
        for (const version of versions) {
            const potentialPath = path.join(ANALYSIS_DIR, version, 'cli.js');
            if (fs.existsSync(potentialPath)) {
                return potentialPath;
            }
        }
    }

    // 4. If not found, download latest from npm
    console.log(`[*] Target cli.js not found. Downloading latest ${PACKAGE_NAME}...`);
    try {
        const version = execSync(`npm view ${PACKAGE_NAME} version`).toString().trim();
        const versionPath = path.join(ANALYSIS_DIR, version);

        if (!fs.existsSync(versionPath)) {
            fs.mkdirSync(versionPath, { recursive: true });
            console.log(`[*] Downloading version ${version} to ${versionPath}...`);
            execSync(`npm pack ${PACKAGE_NAME}@${version}`, { cwd: versionPath });
            const tarball = fs.readdirSync(versionPath).find(f => f.endsWith('.tgz'));
            execSync(`tar -xzf "${tarball}" --strip-components=1`, { cwd: versionPath });
        }

        const downloadedCli = path.join(versionPath, 'cli.js');
        if (fs.existsSync(downloadedCli)) {
            return downloadedCli;
        }
    } catch (err) {
        console.error(`[!] Failed to download bundle: ${err.message}`);
    }

    return null;
}
class CascadeGraph {
    constructor() {
        this.nodes = new Map(); // ChunkName -> { code, tokens, category, neighbors: Set }
        this.internalNameToChunkId = new Map(); // Name -> ChunkName
        this.totalTokens = 0;
    }

    // Helper to extract names defined in a statement
    extractInternalNames(node) {
        const names = [];
        if (node.type === 'VariableDeclaration') {
            node.declarations.forEach(d => {
                if (d.id && d.id.name) names.push(d.id.name);
                if (d.id && d.id.type === 'ObjectPattern') {
                    d.id.properties.forEach(p => {
                        if (p.value && p.value.name) names.push(p.value.name);
                    });
                }
            });
        } else if (node.type === 'FunctionDeclaration' && node.id) {
            names.push(node.id.name);
        } else if (node.type === 'ClassDeclaration' && node.id) {
            names.push(node.id.name);
        }
        return names;
    }

    // Step A: Initial Pass - Group logical chunks
    identifyNodes(ast) {
        console.log(`[*] Phase 1: Identifying Chunks...`);
        const statements = ast.program.body;
        let currentChunkNodes = [];
        let currentTokens = 0;
        let currentCategory = 'unknown';
        let chunkIndex = 1;

        const finalizeChunk = () => {
            if (currentChunkNodes.length === 0) return;
            const chunkName = `chunk${chunkIndex.toString().padStart(3, '0')}`;
            const code = currentChunkNodes.map(n => generate(n, { minified: false }).code).join('\n\n');
            const tokens = encode(code).length;

            this.nodes.set(chunkName, {
                id: chunkIndex,
                name: chunkName,
                code,
                tokens,
                neighbors: new Set(),
                score: 0,
                category: currentCategory
            });

            currentChunkNodes.forEach(node => {
                this.extractInternalNames(node).forEach(name => {
                    this.internalNameToChunkId.set(name, chunkName);
                });
            });

            chunkIndex++;
            currentChunkNodes = [];
            currentTokens = 0;
            currentCategory = 'unknown';
        };

        statements.forEach((node) => {
            const nodeCode = generate(node, { minified: true }).code;
            const nodeTokens = encode(nodeCode).length;
            const hasSignal = SIGNAL_KEYWORDS.some(kw => nodeCode.toLowerCase().includes(kw));
            const nodeCategory = hasSignal ? 'priority' : 'vendor';
            const isImport = node.type === 'ImportDeclaration' ||
                (node.type === 'VariableDeclaration' && nodeCode.includes('require('));

            // Heuristic Split Conditions
            const shouldSplit = currentChunkNodes.length > 0 && (
                isImport ||
                (nodeCategory === 'priority' && currentCategory === 'vendor') ||
                (currentTokens + nodeTokens > 8000) // Slightly larger chunks for context
            );

            if (shouldSplit) {
                finalizeChunk();
            }

            currentChunkNodes.push(node);
            currentTokens += nodeTokens;
            // Upgrade category if priority signal found
            if (nodeCategory === 'priority') currentCategory = 'priority';
            else if (currentCategory === 'unknown') currentCategory = 'vendor';
        });

        finalizeChunk();
    }

    // Step B: Neighbor Detection (The Edges)
    detectNeighbors() {
        console.log(`[*] Phase 2: Neighbor Detection (Building Edges)...`);
        const internalNames = Array.from(this.internalNameToChunkId.keys());

        for (const [chunkName, nodeData] of this.nodes) {
            internalNames.forEach(definedName => {
                const targetChunk = this.internalNameToChunkId.get(definedName);
                if (chunkName === targetChunk) return;

                // Check if this chunk references a name defined in another chunk
                // We use a simple regex-like search but check for word boundaries to avoid partial matches
                if (nodeData.code.includes(definedName)) {
                    // Quick check for word boundary if possible, else just include
                    const regex = new RegExp(`\\b${definedName}\\b`);
                    if (regex.test(nodeData.code)) {
                        nodeData.neighbors.add(targetChunk);
                    }
                }
            });
        }
    }

    // Step C: Markov Chain Analysis
    applyMarkovAnalysis() {
        console.log(`[*] Phase 3: Markov Chain Centrality Calculation...`);
        const names = Array.from(this.nodes.keys());
        const size = names.length;
        if (size === 0) return;

        let scores = new Array(size).fill(1 / size);

        for (let iter = 0; iter < 10; iter++) {
            let nextScores = new Array(size).fill(0);
            for (let i = 0; i < size; i++) {
                const node = this.nodes.get(names[i]);
                const outDegree = node.neighbors.size;

                if (outDegree > 0) {
                    const contribution = scores[i] / outDegree;
                    node.neighbors.forEach(neighborName => {
                        const neighborIdx = names.indexOf(neighborName);
                        if (neighborIdx !== -1) nextScores[neighborIdx] += contribution;
                    });
                } else {
                    for (let j = 0; j < size; j++) nextScores[j] += scores[i] / size;
                }
            }
            scores = nextScores;
        }

        names.forEach((name, i) => {
            this.nodes.get(name).score = scores[i];
        });
    }

    // Phase 4: Re-classify based on aggregated chunk data
    classify() {
        console.log(`[*] Phase 4: Final Classification...`);
        // Categories are already partially set in Phase 1, but we can refine
        // based on score or other metrics if needed.
    }

    saveResults() {
        const chunksDir = path.join(OUTPUT_BASE, 'chunks');
        const metadataDir = path.join(OUTPUT_BASE, 'metadata');

        [chunksDir, metadataDir].forEach(d => {
            if (fs.existsSync(d)) {
                // Clear old results
                fs.readdirSync(d).forEach(f => fs.unlinkSync(path.join(d, f)));
            }
            fs.mkdirSync(d, { recursive: true });
        });

        const metadata = [];

        for (const [name, node] of this.nodes) {
            const fileName = `${name}.js`;
            fs.writeFileSync(path.join(chunksDir, fileName), node.code);

            metadata.push({
                name: node.name,
                file: `chunks/${fileName}`,
                tokens: node.tokens,
                category: node.category,
                centrality: node.score,
                neighborCount: node.neighbors.size,
                outbound: Array.from(node.neighbors)
            });
        }

        fs.writeFileSync(
            path.join(metadataDir, 'graph_map.json'),
            JSON.stringify(metadata, null, 2)
        );

        fs.writeFileSync(
            path.join(metadataDir, 'graph_map.js'),
            `window.GRAPH_DATA = ${JSON.stringify(metadata, null, 2)};`
        );
    }
}

// --- 3. EXECUTION ---
async function run() {
    const targetFile = ensureTargetExists();

    if (!targetFile || !fs.existsSync(targetFile)) {
        console.error(`[!] Error: No input file found.`);
        console.error(`    Checked: CLI arguments, ./cli.js, and ./claude-analysis/*/cli.js`);
        console.error(`    Usage: node analyze.js [path-to-bundle.js]`);
        process.exit(1);
    }

    console.log(`[*] Target identified: ${targetFile}`);
    const code = fs.readFileSync(targetFile, 'utf8');
    const ast = parser.parse(code, {
        sourceType: 'unambiguous',
        plugins: ['jsx', 'typescript'],
        errorRecovery: true
    });

    const graph = new CascadeGraph();
    graph.identifyNodes(ast);
    graph.detectNeighbors();
    graph.applyMarkovAnalysis();
    graph.classify();
    graph.saveResults();

    console.log(`\n[COMPLETE]`);
    console.log(`Analysis saved to: ${OUTPUT_BASE}`);
}

run().catch(console.error);