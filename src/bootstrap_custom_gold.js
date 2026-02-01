const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const CUSTOM_GOLD_DIR = './ml/custom_gold';
const BOOTSTRAP_DIR = path.resolve('./ml/bootstrap_data');
const ANALYSIS_OUTPUT = path.resolve('./cascade_graph_analysis');
const VERSION_DIR_RE = /^\d+\.\d+\.\d+([.-][0-9A-Za-z]+)?$/;
const SKIP_DIRS = new Set(['.git', 'node_modules', 'dist', 'build', 'out', '.next']);

const args = process.argv.slice(2);
const shouldForce = args.includes('--force');
const isVerbose = args.includes('--verbose');
const useBun = args.includes('--bun');
const useEsbuild = args.includes('--esbuild');
const BUNDLERS = (useBun || useEsbuild)
    ? [
        ...(useEsbuild ? ['esbuild'] : []),
        ...(useBun ? ['bun'] : [])
    ]
    : ['esbuild', 'bun'];

const getVersionRoots = () => {
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
};

const walkFiles = (dir, out) => {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (!SKIP_DIRS.has(entry.name)) walkFiles(fullPath, out);
        } else if (
            entry.name.endsWith('.ts') ||
            entry.name.endsWith('.tsx') ||
            entry.name.endsWith('.js') ||
            entry.name.endsWith('.jsx')
        ) {
            out.push(fullPath);
        }
    }
};

const DEFAULT_EXTERNALS = [
    'react',
    'react/jsx-runtime',
    'react/jsx-dev-runtime',
    'ink',
    '@inkjs/ui',
    'lodash-es',
    'lodash',
    'shell-quote',
    'marked',
    'cli-highlight',
    'env-paths',
    'diff',
    'spawn-rx',
    'glob',
    'zod-to-json-schema',
    '@sentry/node',
    '@statsig/js-client',
    '@anthropic-ai/sdk',
    '@anthropic-ai/sdk/shims/node',
    '@anthropic-ai/bedrock-sdk',
    '@anthropic-ai/vertex-sdk',
    '@modelcontextprotocol/sdk/client/index.js',
    '@modelcontextprotocol/sdk/client/stdio.js',
    '@modelcontextprotocol/sdk/client/sse.js',
    '@modelcontextprotocol/sdk/client/websocket.js',
    '@modelcontextprotocol/sdk/server/index.js',
    '@modelcontextprotocol/sdk/server/stdio.js',
    '@modelcontextprotocol/sdk/types.js',
    'micromatch',
    'gray-matter',
    'axios',
    'undici',
    '@azure/msal-node',
    'open',
    'chokidar',
    '@opentelemetry/api',
    '@opentelemetry/resources',
    '@opentelemetry/semantic-conventions',
    '@opentelemetry/sdk-trace-node',
    '@opentelemetry/sdk-metrics',
    '@opentelemetry/sdk-logs',
    '@opentelemetry/exporter-trace-otlp-http',
    '@opentelemetry/exporter-metrics-otlp-http',
    '@opentelemetry/exporter-logs-otlp-http',
    'proper-lockfile',
    'fflate',
    'js-yaml',
    'turndown',
    'ws',
    'cli-table3',
    'figures',
    'ink-link',
    'highlight.js',
    '@commander-js/extra-typings',
    'ansi-escapes',
    'ink-spinner',
    'ink-select-input',
    'ink-text-input',
    'fuse.js'
];

const parseNamedExports = (importClause) => {
    const names = new Set();
    const braceMatch = importClause.match(/\{([^}]+)\}/);
    if (braceMatch) {
        braceMatch[1]
            .split(',')
            .map(part => part.trim())
            .filter(Boolean)
            .forEach(part => {
                const [orig] = part.split(/\s+as\s+/i);
                if (orig) names.add(orig.trim());
            });
    }
    return names;
};

const hasDefaultImport = (importClause) => {
    if (!importClause) return false;
    if (importClause.trim().startsWith('{')) return false;
    if (importClause.includes('* as')) return false;
    return true;
};

const mirrorSources = (rootDir, workDir, files) => {
    const shadowRoot = path.join(workDir, 'shadow');
    const copied = new Map();

    for (const file of files) {
        const rel = path.relative(rootDir, file);
        const dest = path.join(shadowRoot, rel);
        fs.mkdirSync(path.dirname(dest), { recursive: true });
        fs.copyFileSync(file, dest);
        copied.set(file, dest);
    }

    const stubbed = new Set();
    const importRe = /import\s+([^'"]+)\s+from\s+(['"])(\.[^'"]+\.m?js)\2/g;
    const sideEffectRe = /import\s+(['"])(\.[^'"]+\.m?js)\1/g;
    const requireRe = /require\s*\(\s*(['"])(\.[^'"]+\.m?js)\1\s*\)/g;
    const exts = ['.ts', '.tsx', '.jsx'];

    for (const [, shadowPath] of copied) {
        const contents = fs.readFileSync(shadowPath, 'utf8');
        let match;
        const requests = [];
        while ((match = importRe.exec(contents)) !== null) {
            requests.push({ spec: match[3], clause: match[1] });
        }
        while ((match = sideEffectRe.exec(contents)) !== null) {
            requests.push({ spec: match[2], clause: '' });
        }
        while ((match = requireRe.exec(contents)) !== null) {
            requests.push({ spec: match[2], clause: '' });
        }

        for (const req of requests) {
            const spec = req.spec;
            const resolved = path.resolve(path.dirname(shadowPath), spec);
            if (fs.existsSync(resolved)) continue;

            const base = resolved.replace(/\.m?js$/, '');
            const target = exts.map(ext => `${base}${ext}`).find(p => fs.existsSync(p));

            if (stubbed.has(resolved)) continue;
            fs.mkdirSync(path.dirname(resolved), { recursive: true });

            if (target) {
                const relTarget = `./${path.relative(path.dirname(resolved), target).replace(/\\/g, '/')}`;
                const stub = [
                    `export * from ${JSON.stringify(relTarget)};`,
                    `export { default } from ${JSON.stringify(relTarget)};`
                ].join('\n');
                fs.writeFileSync(resolved, stub);
                stubbed.add(resolved);
                continue;
            }

            const names = parseNamedExports(req.clause);
            const lines = [];
            if (hasDefaultImport(req.clause)) {
                lines.push('const __default = {};');
                lines.push('export default __default;');
            }
            for (const name of names) {
                lines.push(`export const ${name} = {};`);
            }
            if (!lines.length) {
                lines.push('export default {};');
            }
            fs.writeFileSync(resolved, lines.join('\n'));
            stubbed.add(resolved);
        }
    }

    return { shadowRoot, copied };
};

const buildExternalFlags = (bundler) => {
    const raw = process.env.CUSTOM_GOLD_EXTERNALS;
    const externals = raw
        ? raw.split(',').map((s) => s.trim()).filter(Boolean)
        : DEFAULT_EXTERNALS;
    if (!externals.length) return '';
    if (bundler === 'bun') {
        return externals.map((ext) => `-e ${ext}`).join(' ');
    }
    return externals.map((ext) => `--external:${ext}`).join(' ');
};

const writeEntryFile = (workDir, files) => {
    const entryPath = path.join(workDir, 'entry.js');
    const lines = [];
    files.forEach((file, idx) => {
        const absPath = path.resolve(file);
        lines.push(`import * as Mod${idx} from ${JSON.stringify(absPath)};`);
        lines.push(`console.log(Mod${idx});`);
    });
    if (!lines.length) {
        lines.push('export {};');
    }
    fs.writeFileSync(entryPath, lines.join('\n'));
    return entryPath;
};

const bootstrapCustomGold = () => {
    const roots = getVersionRoots();
    if (!roots.length) {
        console.warn(`[!] No custom gold directories found at ${CUSTOM_GOLD_DIR}`);
        return;
    }

    if (!fs.existsSync(BOOTSTRAP_DIR)) fs.mkdirSync(BOOTSTRAP_DIR, { recursive: true });

    for (const root of roots) {
        const files = [];
        walkFiles(root.dir, files);

        if (!files.length) {
            console.warn(`[!] No JS/TS files found for custom gold at ${root.dir}`);
            continue;
        }

        for (const bundler of BUNDLERS) {
            const safeName = `${root.label}_${bundler}`;
            const targetStore = path.join(BOOTSTRAP_DIR, `${safeName}_gold_asts.json`);
            if (fs.existsSync(targetStore) && !shouldForce) {
                if (isVerbose) {
                    console.log(`  [SKIP] Found existing fingerprints (${path.basename(targetStore)}).`);
                }
                continue;
            }

            const workDir = path.join(BOOTSTRAP_DIR, '_custom_gold_work', safeName);
            fs.mkdirSync(workDir, { recursive: true });

            console.log(`\n[CUSTOM_GOLD ${root.version || 'root'} | ${bundler}]`);

            const { shadowRoot, copied } = mirrorSources(root.dir, workDir, files);
            const shadowFiles = Array.from(copied.values());
            const entryFile = writeEntryFile(workDir, shadowFiles);
            const bundlePath = path.join(workDir, 'bundled.js');
            const externalFlags = buildExternalFlags(bundler);

            try {
                if (bundler === 'bun') {
                    console.log(`  [+] Bundling with Bun...`);
                    const bunCmd = `bun build "${entryFile}" --minify --target node --outfile "${bundlePath}" ${externalFlags}`;
                    execSync(bunCmd, { stdio: 'inherit' });
                } else {
                    console.log(`  [+] Bundling with esbuild...`);
                    const esbuildCmd = `npx -y esbuild "${entryFile}" --bundle --minify-whitespace --minify-syntax --platform=node --format=esm --target=node18 --log-override:import-is-not-an-export=warning --outfile="${bundlePath}" ${externalFlags}`;
                    execSync(esbuildCmd, { stdio: 'inherit' });
                }

                console.log(`  [+] Fingerprinting structural ASTs (Bootstrap Mode)...`);
                execSync(`node src/analyze.js "${bundlePath}" --version "bootstrap/${safeName}" --is-bootstrap`, { stdio: 'inherit' });

                const resultPath = path.join(ANALYSIS_OUTPUT, 'bootstrap', safeName, 'metadata', 'simplified_asts.json');
                if (fs.existsSync(resultPath)) {
                    fs.copyFileSync(resultPath, targetStore);
                    console.log(`  [OK] Saved custom gold fingerprints to ${path.basename(targetStore)}`);
                } else {
                    console.warn(`  [WARN] Missing simplified_asts.json for ${safeName}`);
                }
            } catch (err) {
                console.error(`  [!] Failed to bootstrap custom gold (${bundler}): ${err.message}`);
            }
        }
    }
};

bootstrapCustomGold();
