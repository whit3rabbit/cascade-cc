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
        } else if (entry.name.endsWith('.ts') || entry.name.endsWith('.js')) {
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
    'ws'
];

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

            const entryFile = writeEntryFile(workDir, files);
            const bundlePath = path.join(workDir, 'bundled.js');
            const externalFlags = buildExternalFlags(bundler);

            try {
                if (bundler === 'bun') {
                    console.log(`  [+] Bundling with Bun...`);
                    const bunCmd = `bun build "${entryFile}" --minify --target node --outfile "${bundlePath}" ${externalFlags}`;
                    execSync(bunCmd, { stdio: 'inherit' });
                } else {
                    console.log(`  [+] Bundling with esbuild...`);
                    const esbuildCmd = `npx -y esbuild "${entryFile}" --bundle --minify-whitespace --minify-syntax --platform=node --format=esm --target=node18 --outfile="${bundlePath}" ${externalFlags}`;
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
