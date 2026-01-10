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
        this.nodes = new Map(); // Name -> { code, tokens, type, neighbors: Set }
        this.adjMatrix = {};    // For Markov calculations
        this.totalTokens = 0;
    }

    // Step A: Initial Pass - Identify all logical chunks (Nodes)
    identifyNodes(ast) {
        console.log(`[*] Phase 1: Identifying Nodes...`);
        ast.program.body.forEach((node, index) => {
            let name = `chunk_${index.toString().padStart(4, '0')}`;

            if (node.type === 'VariableDeclaration' && node.declarations[0].id.name) {
                name = node.declarations[0].id.name;
            } else if (node.type === 'FunctionDeclaration' && node.id) {
                name = node.id.name;
            }

            const code = generate(node, { minified: false }).code;
            const tokens = encode(code).length;

            this.nodes.set(name, {
                id: index,
                name,
                code,
                tokens,
                neighbors: new Set(),
                score: 0, // Markov Centrality
                category: 'unknown'
            });
        });
    }

    // Step B: Neighbor Detection (The Edges)
    detectNeighbors() {
        console.log(`[*] Phase 2: Neighbor Detection (Building Edges)...`);
        const nodeNames = Array.from(this.nodes.keys());

        for (const [name, nodeData] of this.nodes) {
            // Optimization: Only search for names that actually exist in the bundle
            nodeNames.forEach(potentialNeighbor => {
                if (name === potentialNeighbor) return;

                // Simple but effective check for neighbor calling
                // e.g., if code calls 'require_lodash_es()'
                if (nodeData.code.includes(potentialNeighbor)) {
                    nodeData.neighbors.add(potentialNeighbor);
                }
            });
        }
    }

    // Step C: Markov Chain Analysis
    // We treat the bundle as a state machine. 
    // Probability of "stepping" into a neighbor = 1 / total_neighbors.
    applyMarkovAnalysis() {
        console.log(`[*] Phase 3: Markov Chain Centrality Calculation...`);
        const names = Array.from(this.nodes.keys());
        const size = names.length;

        // Initialize scores (Steady State Distribution)
        let scores = new Array(size).fill(1 / size);

        // Run 10 iterations of Power Iteration (PageRank style Markov Chain)
        for (let iter = 0; iter < 10; iter++) {
            let nextScores = new Array(size).fill(0);
            for (let i = 0; i < size; i++) {
                const node = this.nodes.get(names[i]);
                const outDegree = node.neighbors.size;

                if (outDegree > 0) {
                    const contribution = scores[i] / outDegree;
                    node.neighbors.forEach(neighborName => {
                        const neighborIdx = names.indexOf(neighborName);
                        nextScores[neighborIdx] += contribution;
                    });
                } else {
                    // Sink node: redistribute to all
                    for (let j = 0; j < size; j++) nextScores[j] += scores[i] / size;
                }
            }
            scores = nextScores;
        }

        // Map scores back to nodes
        names.forEach((name, i) => {
            this.nodes.get(name).score = scores[i];
        });
    }

    // Step D: Logical Deobfuscation & Classification
    classify() {
        console.log(`[*] Phase 4: Classifier Logic (Vendor vs Logic)...`);
        const avgScore = Array.from(this.nodes.values()).reduce((a, b) => a + b.score, 0) / this.nodes.size;

        for (const [name, node] of this.nodes) {
            const hasSignal = SIGNAL_KEYWORDS.some(kw => node.code.toLowerCase().includes(kw));

            /**
             * HEURISTICS:
             * 1. High Signal = Priority Logic
             * 2. High Markov Score + No Signal = Vendor Library (it's a high-traffic utility)
             * 3. Low Score + No Signal = Boilerplate/Noise
             */
            if (hasSignal) {
                node.category = 'priority';
            } else if (node.score > avgScore * 1.5) {
                node.category = 'vendor';
            } else {
                node.category = 'utility';
            }
        }
    }

    saveResults() {
        const chunksDir = path.join(OUTPUT_BASE, 'chunks');
        const metadataDir = path.join(OUTPUT_BASE, 'metadata');

        [chunksDir, metadataDir].forEach(d => fs.mkdirSync(d, { recursive: true }));

        const metadata = [];

        for (const [name, node] of this.nodes) {
            // Sanitize name for filesystem
            const safeName = name.replace(/[^a-z0-9_]/gi, '_');
            fs.writeFileSync(path.join(chunksDir, `${safeName}.js`), node.code);

            metadata.push({
                name: node.name,
                file: `chunks/${safeName}.js`,
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