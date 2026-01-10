const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { encode } = require('gpt-tokenizer');

/**
 * Calculates Shannon entropy of a string.
 */
function calculateEntropy(str) {
    const len = str.length;
    const frequencies = {};
    for (const char of str) {
        frequencies[char] = (frequencies[char] || 0) + 1;
    }
    return Object.values(frequencies).reduce((sum, f) => {
        const p = f / len;
        return sum - p * Math.log2(p);
    }, 0);
}


const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

// --- CONFIGURATION---
const PACKAGE_NAME = '@anthropic-ai/claude-code';
const ROOT_DIR = './analysis';
const CHUNK_TOKEN_LIMIT = 40000; // Aim for 40k tokens per chunk
const SEARCH_KEYWORDS = ['anthropic', 'claude', 'prompt', 'agent', 'protocol', 'terminal'];

/**
 * 1. Version-Aware Workspace Setup
 */
function getContext() {
    console.log(`[*] Querying NPM for ${PACKAGE_NAME}...`);
    const version = execSync(`npm view ${PACKAGE_NAME} version`).toString().trim();
    const versionPath = path.join(ROOT_DIR, version);

    // Create directory structure
    const dirs = ['chunks', 'priority_critical', 'metadata'];
    dirs.forEach(d => fs.mkdirSync(path.join(versionPath, d), { recursive: true }));

    return { version, versionPath, stateFile: path.join(versionPath, 'state.json') };
}

/**
 * 2. Symbol Extraction (Inspired by your knowledge_graph.py)
 * Uses Babel to find what a chunk actually DOES.
 */
function extractChunkMetadata(code) {
    const symbols = { exports: [], imports: [], hasCriticalLogic: false };
    try {
        const ast = parser.parse(code, { sourceType: 'module', plugins: ['jsx', 'typescript'] });
        traverse(ast, {
            ExportNamedDeclaration({ node }) {
                if (node.declaration?.id) symbols.exports.push(node.declaration.id.name);
                if (node.declaration?.declarations) {
                    node.declaration.declarations.forEach(d => symbols.exports.push(d.id.name));
                }
            },
            CallExpression({ node }) {
                // Look for patterns like process.env or sensitive APIs
                const callee = node.callee.name || "";
                if (SEARCH_KEYWORDS.some(kw => callee.toLowerCase().includes(kw))) {
                    symbols.hasCriticalLogic = true;
                }
            }
        });
    } catch (e) { /* Parsing tiny chunks might fail */ }
    return symbols;
}

/**
 * 3. Token-Aware AST Chunker (Ported logic from your chunker.py)
 */
function chunkLogic(ctx) {
    const unpackedFile = path.join(ctx.versionPath, 'unpacked', 'deobfuscated.js');
    const chunkDir = path.join(ctx.versionPath, 'chunks');

    // Skip if already chunked
    if (fs.readdirSync(chunkDir).length > 0) {
        console.log("[+] Chunks already exist. Skipping AST slice.");
        return;
    }

    console.log(`[*] Slicing 11MB bundle into token-safe chunks...`);
    const code = fs.readFileSync(unpackedFile, 'utf8');
    const ast = parser.parse(code, { sourceType: 'module', errorRecovery: true });

    let currentChunkNodes = [];
    let currentTokens = 0;
    let chunkIdx = 0;

    ast.program.body.forEach((node) => {
        const nodeCode = generate(node).code;
        const nodeTokens = encode(nodeCode).length;

        if (currentTokens + nodeTokens > CHUNK_TOKEN_LIMIT && currentChunkNodes.length > 0) {
            // Save Chunk
            const output = generate({ type: 'File', program: { type: 'Program', body: currentChunkNodes } }).code;
            fs.writeFileSync(path.join(chunkDir, `chunk_${chunkIdx}.js`), output);

            chunkIdx++;
            currentChunkNodes = [node];
            currentTokens = nodeTokens;
        } else {
            currentChunkNodes.push(node);
            currentTokens += nodeTokens;
        }
    });
}

/**
 * 4. Knowledge-Graph Based Prioritization
 */
function prioritize(ctx) {
    const chunkDir = path.join(ctx.versionPath, 'chunks');
    const priorityDir = path.join(ctx.versionPath, 'priority_critical');
    const chunks = fs.readdirSync(chunkDir).filter(f => f.endsWith('.js'));

    console.log(`[*] Building Knowledge Map for ${chunks.length} chunks...`);
    const knowledgeGraph = {};

    chunks.forEach(file => {
        const code = fs.readFileSync(path.join(chunkDir, file), 'utf8');
        const meta = extractChunkMetadata(code);
        const eScore = calculateEntropy(code);

        const isCritical = meta.hasCriticalLogic ||
            SEARCH_KEYWORDS.some(kw => code.toLowerCase().includes(kw)) ||
            eScore > 5.5;

        knowledgeGraph[file] = {
            entropy: eScore,
            exports: meta.exports,
            critical: isCritical
        };

        if (isCritical) {
            console.log(`[!] Critical Chunk Found: ${file} (Exports: ${meta.exports.slice(0, 3)}...)`);
            fs.copyFileSync(path.join(chunkDir, file), path.join(priorityDir, file));
        }
    });

    fs.writeFileSync(path.join(ctx.versionPath, 'knowledge_graph.json'), JSON.stringify(knowledgeGraph, null, 2));
}

/**
 * Main Pipeline Orchestrator
 */
async function main() {
    const ctx = getContext();

    // 1. Download (Idempotent)
    const cliFile = path.join(ctx.versionPath, 'cli.js');
    if (!fs.existsSync(cliFile)) {
        console.log(`[*] Downloading ${PACKAGE_NAME}@${ctx.version}...`);
        execSync(`npm pack ${PACKAGE_NAME}@${ctx.version}`, { cwd: ctx.versionPath });
        const tarball = fs.readdirSync(ctx.versionPath).find(f => f.endsWith('.tgz'));
        execSync(`tar -xzf "${tarball}" --strip-components=1`, { cwd: ctx.versionPath });
    }

    const unpackedDir = path.join(ctx.versionPath, 'unpacked');
    if (!fs.existsSync(unpackedDir) || fs.readdirSync(unpackedDir).length === 0) {
        console.log(`[*] Running Webcrack...`);
        execSync(`npx webcrack "${cliFile}" -o "${unpackedDir}" -f`);
    }

    // 3. Chunk
    chunkLogic(ctx);

    // 4. Prioritize & Knowledge Graph
    prioritize(ctx);

    // 5. Beautify Priority
    console.log(`[*] Beautifying priority files...`);
    execSync(`npx prettier --write "${ctx.versionPath}/priority_critical/*.js"`);

    console.log(`\n[COMPLETE] Knowledge Graph built: ${ctx.versionPath}/knowledge_graph.json`);
    console.log(`[TARGET] Priority logic ready for Gemini in: ${ctx.versionPath}/priority_critical`);
}

main();