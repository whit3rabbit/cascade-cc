const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

const KB_PATH = path.resolve('knowledge_base.json');
const BACKUP_PATH = `${KB_PATH}.bak`;
const CUSTOM_GOLD_DIR = path.resolve('ml/custom_gold');
const VERSION_DIR_RE = /^\d+\.\d+\.\d+([.-][0-9A-Za-z]+)?$/;
const SKIP_DIRS = new Set(['.git', 'node_modules', 'dist', 'build', 'out', '.next']);

const PARSER_PLUGINS = [
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
];

const ERROR_CALLEES = new Set([
    'Error',
    'TypeError',
    'RangeError',
    'SyntaxError',
    'ReferenceError',
    'invariant',
    'assert',
    'ensure',
    'fail',
    'panic'
]);

function getVersionRoots() {
    if (!fs.existsSync(CUSTOM_GOLD_DIR)) return [];
    const entries = fs.readdirSync(CUSTOM_GOLD_DIR, { withFileTypes: true });
    const versionDirs = entries.filter(
        (entry) => entry.isDirectory() && VERSION_DIR_RE.test(entry.name)
    );

    if (versionDirs.length > 0) {
        return versionDirs.map((entry) => ({
            version: entry.name,
            dir: path.join(CUSTOM_GOLD_DIR, entry.name)
        }));
    }

    const srcDir = path.join(CUSTOM_GOLD_DIR, 'src');
    if (fs.existsSync(srcDir) && fs.statSync(srcDir).isDirectory()) {
        return [{
            version: null,
            dir: srcDir
        }];
    }

    return [{
        version: null,
        dir: CUSTOM_GOLD_DIR
    }];
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
        } else if (/\.(ts|tsx|js|jsx)$/.test(file.name)) {
            const code = fs.readFileSync(fullPath, 'utf8');
            fileHandler(fullPath, code);
        }
    }
}

function normalizeString(str) {
    return String(str || '').replace(/\s+/g, ' ').trim();
}

function extractWordTokens(text) {
    const matches = text.match(/[A-Za-z0-9_./-]{4,}/g);
    return matches ? matches : [];
}

function isUsefulString(str) {
    if (!str) return false;
    const trimmed = normalizeString(str);
    if (trimmed.length < 4 || trimmed.length > 120) return false;
    if (!/[A-Za-z]/.test(trimmed)) return false;
    if (trimmed.includes('\n')) return false;
    return true;
}

function parseFile(code) {
    const exports = new Set();
    const strings = new Set();
    const errorStrings = new Set();

    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: PARSER_PLUGINS
        });

        traverse(ast, {
            ExportNamedDeclaration(path) {
                const decl = path.node.declaration;
                if (decl && decl.id && decl.id.name) exports.add(decl.id.name);
                if (decl && decl.declarations) {
                    decl.declarations.forEach((d) => {
                        if (d.id && d.id.name) exports.add(d.id.name);
                    });
                }
                if (path.node.specifiers) {
                    path.node.specifiers.forEach((s) => {
                        if (s.exported && s.exported.name) exports.add(s.exported.name);
                    });
                }
            },
            ExportDefaultDeclaration(path) {
                const decl = path.node.declaration;
                if (decl && decl.id && decl.id.name) {
                    exports.add(decl.id.name);
                } else {
                    exports.add('default');
                }
            },
            StringLiteral(path) {
                const val = normalizeString(path.node.value);
                if (isUsefulString(val)) strings.add(val);
            },
            TemplateLiteral(path) {
                path.node.quasis.forEach((q) => {
                    const val = normalizeString(q.value.cooked || q.value.raw || '');
                    if (isUsefulString(val)) strings.add(val);
                });
            },
            JSXText(path) {
                const val = normalizeString(path.node.value);
                if (isUsefulString(val)) strings.add(val);
            },
            ThrowStatement(path) {
                const arg = path.node.argument;
                if (!arg) return;
                if (arg.type === 'NewExpression' || arg.type === 'CallExpression') {
                    const callee = arg.callee;
                    const calleeName = callee && callee.name ? callee.name : null;
                    if (calleeName && ERROR_CALLEES.has(calleeName)) {
                        const firstArg = arg.arguments && arg.arguments[0];
                        if (firstArg && firstArg.type === 'StringLiteral') {
                            const val = normalizeString(firstArg.value);
                            if (isUsefulString(val)) errorStrings.add(val);
                        }
                    }
                }
            },
            CallExpression(path) {
                const callee = path.node.callee;
                if (!callee || callee.type !== 'Identifier') return;
                if (!ERROR_CALLEES.has(callee.name)) return;
                const firstArg = path.node.arguments && path.node.arguments[0];
                if (firstArg && firstArg.type === 'StringLiteral') {
                    const val = normalizeString(firstArg.value);
                    if (isUsefulString(val)) errorStrings.add(val);
                }
            }
        });
    } catch (err) {
        // Keep going; a single parse error shouldn't break the generator.
    }

    return { exports, strings, errorStrings };
}

function buildProjectStructure(entries) {
    const root = {};

    function ensureDir(tree, name) {
        if (!tree[name]) {
            tree[name] = { description: 'Auto-generated from custom_gold' };
        } else if (typeof tree[name] !== 'object') {
            tree[name] = { description: 'Auto-generated from custom_gold' };
        } else if (!tree[name].description) {
            tree[name].description = 'Auto-generated from custom_gold';
        }
        return tree[name];
    }

    entries.forEach((entry) => {
        const parts = entry.relPath.split(path.sep);
        if (!parts.length) return;

        let cursor = ensureDir(root, 'src');
        for (let i = 1; i < parts.length; i++) {
            const part = parts[i];
            const isFile = i === parts.length - 1;
            if (isFile) {
                if (!cursor.files) cursor.files = [];
                if (!cursor.files.includes(part)) cursor.files.push(part);
            } else {
                cursor = ensureDir(cursor, part);
            }
        }
    });

    return root;
}

function main() {
    if (fs.existsSync(KB_PATH)) {
        fs.copyFileSync(KB_PATH, BACKUP_PATH);
        console.log(`[*] Backed up existing KB to ${BACKUP_PATH}`);
    }

    const previousKb = fs.existsSync(BACKUP_PATH)
        ? JSON.parse(fs.readFileSync(BACKUP_PATH, 'utf8'))
        : {};

    const roots = getVersionRoots();
    if (!roots.length) {
        console.error(`[!] No custom_gold sources found at ${CUSTOM_GOLD_DIR}`);
        process.exit(1);
    }

    const fileInfos = [];
    const globalTokenFreq = new Map();

    roots.forEach((root) => {
        walk(root.dir, (fullPath, code) => {
            const relPath = path.relative(root.dir, fullPath);
            const parsed = parseFile(code);

            const base = path.basename(relPath).replace(/\.[^.]+$/, '');
            const exports = Array.from(parsed.exports).filter(Boolean);
            const suggestedName = exports.find((n) => n !== 'default') || exports[0] || base;

            const tokens = new Set();
            tokens.add(base);
            if (suggestedName) tokens.add(suggestedName);
            parsed.strings.forEach((s) => {
                extractWordTokens(s).forEach((t) => tokens.add(t));
            });

            tokens.forEach((t) => {
                const key = t.toLowerCase();
                globalTokenFreq.set(key, (globalTokenFreq.get(key) || 0) + 1);
            });

            fileInfos.push({
                relPath,
                base,
                suggestedName,
                tokens: Array.from(tokens),
                strings: Array.from(parsed.strings),
                errorStrings: Array.from(parsed.errorStrings)
            });
        });
    });

    const fileAnchors = [];
    const nameHints = [];
    const errorAnchors = [];

    fileInfos.forEach((info) => {
        const tokens = info.tokens
            .filter((t) => t.length >= 4 && t.length <= 40)
            .filter((t) => !/^\d+$/.test(t));

        tokens.sort((a, b) => {
            const fa = globalTokenFreq.get(a.toLowerCase()) || 0;
            const fb = globalTokenFreq.get(b.toLowerCase()) || 0;
            if (fa !== fb) return fa - fb;
            if (a.length !== b.length) return b.length - a.length;
            return a.localeCompare(b);
        });

        const triggerKeywords = [];
        for (const t of tokens) {
            if (!triggerKeywords.includes(t)) triggerKeywords.push(t);
            if (triggerKeywords.length >= 8) break;
        }

        const suggestedPath = `\`src/${info.relPath.replace(/\\/g, '/')}\``;
        const suggestedName = `\`${info.suggestedName}\``;
        const description = `Auto-generated anchor for ${info.relPath}`;

        fileAnchors.push({
            suggested_path: suggestedPath,
            suggested_name: suggestedName,
            description,
            trigger_keywords: triggerKeywords
        });

        const hintStrings = info.strings
            .filter((s) => s.length >= 8 && s.length <= 120)
            .sort((a, b) => b.length - a.length)
            .slice(0, 5);

        hintStrings.forEach((s) => {
            nameHints.push({
                suggested_name: suggestedName,
                logic_anchor: s
            });
        });

        info.errorStrings.forEach((s) => {
            errorAnchors.push({
                role: 'AUTO_ERROR',
                content: s
            });
        });
    });

    const mergedErrorAnchors = [];
    const seenErrorContent = new Set();
    [...(previousKb.error_anchors || []), ...errorAnchors].forEach((e) => {
        if (!e || !e.content) return;
        if (seenErrorContent.has(e.content)) return;
        seenErrorContent.add(e.content);
        mergedErrorAnchors.push(e);
    });

    const kb = {
        project_structure: buildProjectStructure(fileInfos),
        file_anchors: fileAnchors,
        name_hints: nameHints,
        error_anchors: mergedErrorAnchors,
        structural_anchors: previousKb.structural_anchors || [],
        known_packages: previousKb.known_packages || []
    };

    fs.writeFileSync(KB_PATH, JSON.stringify(kb, null, 2));
    console.log(`[*] Wrote new KB to ${KB_PATH}`);
}

main();
