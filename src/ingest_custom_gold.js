const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const { simplifyAST } = require('./analyze'); // Reuse your existing AST simplifier

const CUSTOM_GOLD_DIR = './ml/custom_gold';
const BOOTSTRAP_DATA_DIR = './ml/bootstrap_data';

// Ensure bootstrap directory exists
if (!fs.existsSync(BOOTSTRAP_DATA_DIR)) {
    fs.mkdirSync(BOOTSTRAP_DATA_DIR, { recursive: true });
}

function ingest() {
    const goldAsts = {};

    function walk(dir) {
        if (!fs.existsSync(dir)) return;

        const files = fs.readdirSync(dir);
        for (const file of files) {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                walk(fullPath);
            } else if (file.endsWith('.ts') || file.endsWith('.js')) {
                const code = fs.readFileSync(fullPath, 'utf8');
                const relPath = path.relative(CUSTOM_GOLD_DIR, fullPath);

                console.log(`[*] Ingesting: ${relPath}`);

                try {
                    const ast = parser.parse(code, {
                        sourceType: 'module',
                        plugins: ['typescript', 'jsx', 'decorators-legacy', 'classProperties']
                    });

                    // Treat the whole file as a logic chunk or split by function
                    // Here we simplify the whole program body
                    const simplified = ast.program.body.map(node => simplifyAST(node));

                    const chunkName = relPath.replace(/[\/\.]/g, '_');
                    goldAsts[chunkName] = {
                        ast: simplified,
                        proposedPath: relPath, // Crucial for classification
                        category: 'founder'    // Label as proprietary logic
                    };
                } catch (e) {
                    console.error(`[!] Failed to parse ${file}: ${e.message}`);
                }
            }
        }
    }

    walk(CUSTOM_GOLD_DIR);

    const outputPath = path.join(BOOTSTRAP_DATA_DIR, 'custom_claude_gold_asts.json');
    fs.writeFileSync(outputPath, JSON.stringify(goldAsts, null, 2));
    console.log(`[+] Saved ${Object.keys(goldAsts).length} custom logic patterns to ${outputPath}`);

    // NEW: Sync to cascade_graph_analysis/bootstrap for vectorization/registry sync
    const REGISTRY_BOOTSTRAP_DIR = './cascade_graph_analysis/bootstrap/custom_claude_gold';
    if (!fs.existsSync(REGISTRY_BOOTSTRAP_DIR)) {
        fs.mkdirSync(REGISTRY_BOOTSTRAP_DIR, { recursive: true });
    }

    // Recursive copy
    function copyDir(src, dest) {
        if (!fs.existsSync(src)) return;
        const entries = fs.readdirSync(src, { withFileTypes: true });
        if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });

        for (const entry of entries) {
            const srcPath = path.join(src, entry.name);
            const destPath = path.join(dest, entry.name);
            if (entry.isDirectory()) {
                copyDir(srcPath, destPath);
            } else {
                fs.copyFileSync(srcPath, destPath);
            }
        }
    }

    console.log(`[*] Syncing files to ${REGISTRY_BOOTSTRAP_DIR} for registry anchoring...`);
    copyDir(CUSTOM_GOLD_DIR, REGISTRY_BOOTSTRAP_DIR);
    console.log(`[+] Synced source files.`);
}

ingest();
