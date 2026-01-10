const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { entropy } = require('shannon-entropy-js');

// Babel tools for manual chunking
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

// --- CONFIGURATION ---
const PACKAGE_NAME = '@anthropic-ai/claude-code';
const ROOT_ANALYSIS_DIR = './analysis';
const SEARCH_KEYWORDS = ['anthropic', 'claude', 'claude code', 'prompt', 'tool', 'agent'];
const CHUNK_SIZE_THRESHOLD = 100000; // ~100KB per manual chunk

/**
 * 1. Setup & Get Context
 */
function getTargetContext() {
    console.log(`[*] Checking for latest version of ${PACKAGE_NAME}...`);
    const version = execSync(`npm view ${PACKAGE_NAME} version`).toString().trim();
    const safeName = PACKAGE_NAME.replace('@', '').replace('/', '-');
    const versionPath = path.join(ROOT_ANALYSIS_DIR, safeName, version);
    return { version, versionPath };
}

/**
 * 2. Manual AST Chunker
 * This is the "Magic" step. It takes a giant JS file and slices it into
 * manageable files based on top-level definitions.
 */
function manualAstChunking(inputFilePath, outputDir) {
    console.log(`[*] Hard Chunking: Parsing 11MB AST (This requires memory)...`);
    const code = fs.readFileSync(inputFilePath, 'utf8');
    
    const ast = parser.parse(code, {
        sourceType: 'module',
        plugins: ['jsx', 'typescript'], // Cover all bases
        errorRecovery: true
    });

    let currentChunkNodes = [];
    let currentChunkSize = 0;
    let chunkCount = 0;

    const saveChunk = () => {
        if (currentChunkNodes.length === 0) return;
        chunkCount++;
        const chunkAst = { type: 'File', program: { type: 'Program', body: currentChunkNodes } };
        const output = generate(chunkAst, { minified: false }).code;
        fs.writeFileSync(path.join(outputDir, `ast_chunk_${chunkCount}.js`), output);
        currentChunkNodes = [];
        currentChunkSize = 0;
    };

    // We iterate through the top-level statements of the program
    ast.program.body.forEach((node) => {
        currentChunkNodes.push(node);
        // Estimate size based on node location in original source
        if (node.end && node.start) {
            currentChunkSize += (node.end - node.start);
        }

        if (currentChunkSize > CHUNK_SIZE_THRESHOLD) {
            saveChunk();
        }
    });

    saveChunk();
    console.log(`[+] Created ${chunkCount} manual chunks from massive bundle.`);
}

/**
 * 3. Main Workflow
 */
async function run() {
    const ctx = getTargetContext();
    const { versionPath } = ctx;

    // A. Setup Folder
    if (!fs.existsSync(versionPath)) {
        fs.mkdirSync(versionPath, { recursive: true });
        console.log(`[*] Downloading...`);
        execSync(`npm pack ${PACKAGE_NAME}@${ctx.version}`, { cwd: versionPath });
        const tarball = fs.readdirSync(versionPath).find(f => f.endsWith('.tgz'));
        execSync(`tar -xzf "${tarball}" --strip-components=1`, { cwd: versionPath });
    }

    // B. Unpack (Webcrack)
    const binPath = path.join(versionPath, 'cli.js');
    const unpackedDir = path.join(versionPath, 'unpacked');
    if (!fs.existsSync(unpackedDir)) {
        console.log(`[*] Running Webcrack...`);
        execSync(`npx webcrack "${binPath}" -o "${unpackedDir}"`);
    }

    // C. Detect if Webcrack failed to chunk
    const extractedFiles = fs.readdirSync(unpackedDir).filter(f => f.endsWith('.js'));
    const largestFile = extractedFiles
        .map(f => ({ name: f, size: fs.statSync(path.join(unpackedDir, f)).size }))
        .sort((a, b) => b.size - a.size)[0];

    const chunkDir = path.join(versionPath, 'chunks');
    if (!fs.existsSync(chunkDir)) fs.mkdirSync(chunkDir);

    // If the largest file is > 1MB, Webcrack didn't split the logic. Let's do it manually.
    if (largestFile.size > 1000000) {
        console.log(`[!] Logic still bundled in ${largestFile.name}. Slicing AST...`);
        manualAstChunking(path.join(unpackedDir, largestFile.name), chunkDir);
    }

    // D. Priority Scan
    const priorityDir = path.join(versionPath, 'priority_critical');
    if (!fs.existsSync(priorityDir)) fs.mkdirSync(priorityDir);

    const chunkFiles = fs.readdirSync(chunkDir);
    chunkFiles.forEach(file => {
        const code = fs.readFileSync(path.join(chunkDir, file), 'utf8');
        const hasKeyword = SEARCH_KEYWORDS.some(kw => code.toLowerCase().includes(kw));
        
        if (hasKeyword) {
            console.log(`[!] Found keywords in ${file}. Moving to priority.`);
            fs.copyFileSync(path.join(chunkDir, file), path.join(priorityDir, file));
        }
    });

    // E. Beautify Priority
    console.log(`[*] Beautifying priority chunks...`);
    try {
        execSync(`npx prettier --write "${priorityDir}/*.js"`);
    } catch (e) {}

    console.log(`\n[SUCCESS] Sliced bundle into logical chunks.`);
    console.log(`[DIR] ${priorityDir}`);
}

run();