const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const { simplifyAST } = require('./analyze'); // Reuse your existing AST simplifier

const CUSTOM_GOLD_DIR = './ml/custom_gold';
const BOOTSTRAP_DATA_DIR = './ml/bootstrap_data';
const VERSION_DIR_RE = /^\d+\.\d+\.\d+([.-][0-9A-Za-z]+)?$/;
const SKIP_DIRS = new Set(['.git', 'node_modules', 'dist', 'build', 'out', '.next']);

// Ensure bootstrap directory exists
if (!fs.existsSync(BOOTSTRAP_DATA_DIR)) {
    fs.mkdirSync(BOOTSTRAP_DATA_DIR, { recursive: true });
}

function getVersionRoots() {
    if (!fs.existsSync(CUSTOM_GOLD_DIR)) return [];
    const entries = fs.readdirSync(CUSTOM_GOLD_DIR, { withFileTypes: true });
    const versionDirs = entries.filter(
        (entry) => entry.isDirectory() && VERSION_DIR_RE.test(entry.name)
    );

    if (versionDirs.length > 0) {
        return versionDirs.map((entry) => ({
            version: entry.name,
            dir: path.join(CUSTOM_GOLD_DIR, entry.name),
            label: `custom_claude_gold_v${entry.name.replace(/\./g, '_')}`
        }));
    }

    const srcDir = path.join(CUSTOM_GOLD_DIR, 'src');
    if (fs.existsSync(srcDir) && fs.statSync(srcDir).isDirectory()) {
        return [{
            version: null,
            dir: srcDir,
            label: 'custom_claude_gold'
        }];
    }

    return [{
        version: null,
        dir: CUSTOM_GOLD_DIR,
        label: 'custom_claude_gold'
    }];
}

function getNodeName(node) {
    if (!node) return '';
    if (node.id && node.id.name) return node.id.name;
    if (node.declaration && node.declaration.id && node.declaration.id.name) return node.declaration.id.name;
    if (node.type === 'ExportDefaultDeclaration') return 'default';
    if (node.type === 'VariableDeclaration' && node.declarations && node.declarations[0]) {
        const decl = node.declarations[0];
        if (decl.id && decl.id.name) return decl.id.name;
    }
    return '';
}

function ingest() {
    const versionRoots = getVersionRoots();
    if (!versionRoots.length) {
        console.warn(`[!] No custom gold directories found at ${CUSTOM_GOLD_DIR}`);
        return;
    }

    function walk(dir, fileHandler) {
        if (!fs.existsSync(dir)) return;

        const files = fs.readdirSync(dir, { withFileTypes: true });
        for (const file of files) {
            const fullPath = path.join(dir, file.name);
            if (file.isDirectory()) {
                if (!SKIP_DIRS.has(file.name)) {
                    walk(fullPath, fileHandler);
                }
            } else if (file.name.endsWith('.ts') || file.name.endsWith('.js')) {
                const code = fs.readFileSync(fullPath, 'utf8');
                fileHandler(fullPath, code);
            }
        }
    }

    function ingestRoot(root) {
        const goldAsts = {};

        walk(root.dir, (fullPath, code) => {
            const relPath = path.relative(root.dir, fullPath);

            console.log(`[*] Ingesting: ${root.version ? `${root.version}/` : ''}${relPath}`);

            try {
                const ast = parser.parse(code, {
                    sourceType: 'module',
                    plugins: [
                        'typescript',
                        'jsx',
                        'decorators-legacy',
                        'classProperties',
                        'classPrivateProperties',
                        'classPrivateMethods',
                        'privateIn',
                        'topLevelAwait',
                        'importAssertions',
                        'importAttributes',
                        'dynamicImport',
                        'optionalChaining',
                        'nullishCoalescingOperator',
                        'exportDefaultFrom',
                        'exportNamespaceFrom'
                    ]
                });

                const baseName = relPath.replace(/[\/\\.]/g, '_');
                const body = ast.program.body || [];
                const chunks = [];

                body.forEach((node, idx) => {
                    const targetNode = node.declaration || node;
                    const simplified = simplifyAST(targetNode);
                    if (!simplified) return;
                    const nodeName = getNodeName(node);
                    const chunkName = `${baseName}_n${idx}${nodeName ? `_${nodeName}` : ''}`;
                    chunks.push({ chunkName, ast: [simplified] });
                });

                if (!chunks.length) {
                    const simplified = body.map((node) => simplifyAST(node)).filter(Boolean);
                    if (simplified.length) {
                        chunks.push({ chunkName: baseName, ast: simplified });
                    }
                }

                chunks.forEach((chunk) => {
                    goldAsts[chunk.chunkName] = {
                        ast: chunk.ast,
                        proposedPath: relPath,
                        category: 'founder'
                    };
                });
            } catch (e) {
                console.error(`[!] Failed to parse ${relPath}: ${e.message}`);
            }
        });

        const outputPath = path.join(BOOTSTRAP_DATA_DIR, `${root.label}_gold_asts.json`);
        fs.writeFileSync(outputPath, JSON.stringify(goldAsts, null, 2));
        console.log(`[+] Saved ${Object.keys(goldAsts).length} custom logic patterns to ${outputPath}`);

        // Emit simplified_asts.json for vectorization/registry build
        const registryBaseDir = path.join('./cascade_graph_analysis/bootstrap', root.label);
        const registryMetaDir = path.join(registryBaseDir, 'metadata');
        if (!fs.existsSync(registryMetaDir)) {
            fs.mkdirSync(registryMetaDir, { recursive: true });
        }
        const simplifiedAstsPath = path.join(registryMetaDir, 'simplified_asts.json');
        fs.writeFileSync(simplifiedAstsPath, JSON.stringify(goldAsts, null, 2));
        console.log(`[+] Wrote simplified ASTs to ${simplifiedAstsPath}`);

        // Sync to cascade_graph_analysis/bootstrap for vectorization/registry sync
        if (!fs.existsSync(registryBaseDir)) {
            fs.mkdirSync(registryBaseDir, { recursive: true });
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
                    if (!SKIP_DIRS.has(entry.name)) {
                        copyDir(srcPath, destPath);
                    }
                } else {
                    fs.copyFileSync(srcPath, destPath);
                }
            }
        }

        console.log(`[*] Syncing files to ${registryBaseDir} for registry anchoring...`);
        copyDir(root.dir, registryBaseDir);
        console.log(`[+] Synced source files.`);
    }

    versionRoots.forEach(ingestRoot);
}

ingest();
