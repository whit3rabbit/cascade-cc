const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

/**
 * Updated Library Manifest (2026 Stable Versions)
 * Expanded to include all critical dependencies for deobfuscation mapping.
 * 
 * NOTE ON VERSION DRIFT: Multiple versions of the same library are included to 
 * account for "version drift". This provides the model with a broader range of 
 * "Gold Standard" logic fingerprints, allowing it to correctly identify core 
 * library logic even if the target codebase uses a slightly older or newer version.
 */
const LIBS = [
    // --- The "Brain" and UI Framework ---
    { name: 'zod', version: '4.3.5' },
    { name: 'zod', version: '4.2.1' },
    { name: 'zod', version: '3.23.8' },
    { name: 'react', version: '19.2.3' },
    { name: 'react', version: '18.3.1' },
    { name: 'ink', version: '6.6.0', peerDeps: ['react@18.3.1'] },
    { name: '@inkjs/ui', version: '2.0.0', peerDeps: ['react@18.3.1', 'ink@6.6.0'] },
    { name: 'ink-text-input', version: '6.0.0', peerDeps: ['react@18.3.1', 'ink@6.6.0'] },
    { name: 'ink-select-input', version: '6.2.0', peerDeps: ['react@18.3.1', 'ink@6.6.0'] },
    { name: 'ink-link', version: '5.0.0', peerDeps: ['react@18.3.1', 'ink@6.6.0'] },
    { name: 'ink-spinner', version: '5.0.0', peerDeps: ['react@18.3.1', 'ink@6.6.0'] },

    // --- Primary SDKs (Anthropic, AWS, GCP, MCP) ---
    { name: '@anthropic-ai/sdk', version: '0.71.2' },
    { name: '@anthropic-ai/sdk', version: '0.40.0' },
    { name: '@anthropic-ai/bedrock-sdk', version: '0.26.0' },
    { name: '@anthropic-ai/vertex-sdk', version: '0.14.0' },
    { name: '@aws-sdk/client-bedrock', version: '3.962.0' },
    { name: '@aws-sdk/client-bedrock-runtime', version: '3.962.0' },
    { name: '@aws-sdk/client-s3', version: '3.958.0' },
    { name: '@aws-sdk/client-sts', version: '3.958.0' },
    { name: '@aws-sdk/credential-providers', version: '3.958.0' },
    {
        name: '@modelcontextprotocol/sdk',
        version: '1.25.1',
        peerDeps: ['zod@3.23.8'],
        imports: ['@modelcontextprotocol/sdk/client', '@modelcontextprotocol/sdk/server', '@modelcontextprotocol/sdk/validation']
    },

    // --- Common Utilities & Data Handling ---
    { name: 'axios', version: '1.13.2' },
    { name: 'axios', version: '1.7.9' },
    { name: 'axios', version: '0.27.2' },
    { name: 'lodash-es', version: '4.17.22' },
    { name: 'lodash-es', version: '4.17.21' },
    { name: 'ajv', version: '8.17.1' },
    { name: 'date-fns', version: 'latest' },
    { name: 'js-yaml', version: '4.1.1' },
    { name: 'dotenv', version: '17.2.3' },
    { name: 'uuid', version: 'latest' },
    { name: 'semver', version: '7.7.3' },
    { name: 'lru-cache', version: '11.2.4' },
    { name: 'memoize', version: '10.2.0' },
    { name: 'micromatch', version: '4.0.8' },
    { name: 'mime-types', version: '3.0.2' },

    // --- CLI & Terminal UX ---
    { name: 'execa', version: '9.6.1' },
    { name: 'chalk', version: '5.6.2' },
    { name: 'commander', version: '14.0.2' },
    { name: 'ansi-escapes', version: '7.2.0' },
    { name: 'ansi-styles', version: '6.2.3' },
    { name: 'cli-highlight', version: '2.1.11' },
    { name: 'cli-table3', version: '0.6.5' },
    { name: 'figures', version: '6.1.0' },
    { name: 'is-unicode-supported', version: '2.1.0' },
    { name: 'string-width', version: '8.1.0' },
    { name: 'wrap-ansi', version: '9.0.2' },
    { name: 'supports-hyperlinks', version: '4.4.0' },
    { name: 'wcwidth', version: '1.0.1' },

    // --- Observability & Analytics ---
    { name: '@opentelemetry/api', version: '1.9.0' },
    { name: '@opentelemetry/core', version: '2.2.0' },
    { name: '@opentelemetry/sdk-trace-node', version: '2.2.0' },
    { name: '@segment/analytics-node', version: 'latest' },
    { name: '@sentry/node', version: '10.32.1' },
    { name: 'statsig-js', version: '5.1.0' },

    // --- Logic & Parsing (Tree-sitter, etc) ---
    { name: 'tree-sitter', version: '0.21.0' },
    { name: 'tree-sitter-typescript', version: '0.23.2' },
    { name: 'web-tree-sitter', version: '0.26.3' },
    { name: 'jsdom', version: '27.4.0' },
    { name: 'parse5', version: 'latest' },
    { name: 'domino', version: 'latest' },
    { name: 'turndown', version: '7.2.2' },
    { name: 'marked', version: '17.0.1' },
    { name: 'gray-matter', version: '4.0.3' },

    // --- System & Hardware ---
    { name: 'sharp', version: '0.34.5' },
    { name: '@resvg/resvg-js', version: '2.6.2' },
    { name: 'pdf-parse', version: '2.4.5' },
    { name: 'chokidar', version: '5.0.0' },
    { name: 'proper-lockfile', version: '4.1.2' },
    { name: 'open', version: '11.0.0' },

    // --- Network & Protocol ---
    { name: 'ws', version: '8.18.3' },
    { name: 'https-proxy-agent', version: '7.0.6' },
    { name: 'abort-controller', version: '3.0.0' },

    // --- Others ---
    { name: 'diff', version: '8.0.2' },
    { name: 'fflate', version: 'latest' },
    { name: 'fuse.js', version: '7.1.0' },
    { name: 'grapheme-splitter', version: '1.0.4' },
    { name: 'highlight.js', version: '11.11.1' },
    { name: 'html-entities', version: 'latest' },
    { name: 'localforage', version: 'latest' },
    { name: 'ordered-map', version: '0.1.0' },
    { name: 'plist', version: '3.1.0' },
    { name: 'shell-quote', version: '1.8.3' },
    { name: 'tslib', version: 'latest' },
    { name: 'uri-js', version: 'latest' },
    { name: 'word-wrap', version: '1.2.5' },
    { name: 'xmlbuilder2', version: '3.1.1' },
    { name: 'xss', version: 'latest' },
    { name: 'yoga-layout-prebuilt', version: '1.10.0' }
];

const BOOTSTRAP_DIR = path.resolve('./ml/bootstrap_data');
const ANALYSIS_OUTPUT = path.resolve('./cascade_graph_analysis');

async function bootstrap() {
    const isCI = process.env.CI === 'true' || process.argv.includes('--yes');

    if (!isCI) {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        const answer = await new Promise(resolve => {
            rl.question('[?] This will download and analyze multiple libraries. Proceed? (y/N) ', resolve);
        });

        rl.close();
        if (answer.toLowerCase() !== 'y' && answer.toLowerCase() !== 'yes') {
            console.log('[!] Bootstrap aborted.');
            process.exit(0);
        }
    }

    console.log(`[*] STARTING ISOLATED VERSION DRIFT BOOTSTRAP`);

    if (!fs.existsSync(BOOTSTRAP_DIR)) fs.mkdirSync(BOOTSTRAP_DIR, { recursive: true });

    for (const lib of LIBS) {
        const libSafeName = `${lib.name.replace(/[@\/]/g, '_')}_v${lib.version.replace(/\./g, '_')}`;
        const libWorkDir = path.join(BOOTSTRAP_DIR, libSafeName);

        console.log(`\n[${lib.name.toUpperCase()} @ ${lib.version}]`);
        if (!fs.existsSync(libWorkDir)) fs.mkdirSync(libWorkDir, { recursive: true });

        try {
            // 1. Isolated Install for this specific version
            console.log(`  [+] Installing ${lib.name}@${lib.version}...`);
            const installCmd = `npm install ${lib.name}@${lib.version} ${lib.peerDeps ? lib.peerDeps.join(' ') : ''} --no-save --prefix "${libWorkDir}" --legacy-peer-deps --install-links`;
            execSync(installCmd, { stdio: 'ignore' });

            // 2. Create Entry Point (mimic bundling)
            const entryFile = path.join(libWorkDir, 'entry.js');
            let importCode = '';
            if (lib.imports) {
                lib.imports.forEach((imp, i) => {
                    importCode += `import * as Lib${i} from '${imp}';\nconsole.log(Lib${i});\n`;
                });
            } else {
                importCode = `import * as Lib from '${lib.name}';\nconsole.log(Lib);\n`;
            }
            fs.writeFileSync(entryFile, importCode);

            // 3. Bundle with ESBUILD
            const bundlePath = path.join(libWorkDir, 'bundled.js');
            console.log(`  [+] Bundling with esbuild...`);

            const nodePath = path.join(libWorkDir, 'node_modules');
            const baseExternals = [
                'sharp',
                'tree-sitter',
                'tree-sitter-typescript',
                'react-devtools-core',
                'fsevents',
                'react',
                'react/jsx-runtime',
                'react/jsx-dev-runtime',
                'ink',
                '@opentelemetry/api',
                '@opentelemetry/sdk-trace-node'
            ];

            const getPackageName = (spec) => {
                if (!spec) return spec;
                if (spec.startsWith('@')) {
                    const parts = spec.split('@');
                    return parts.length >= 3 ? `@${parts[1]}` : spec;
                }
                return spec.split('@')[0];
            };

            const peerDeps = (lib.peerDeps || []).map(getPackageName);
            const externals = [...new Set([...baseExternals, ...peerDeps])]
                .filter(ext => ext !== lib.name && !lib.name.startsWith(`${ext}/`));

            const externalFlags = externals.map(ext => `--external:${ext}`).join(' ');
            execSync(`NODE_PATH="${nodePath}" npx -y esbuild "${entryFile}" --bundle --minify-whitespace --minify-syntax --platform=node --format=esm --target=node18 --outfile="${bundlePath}" --loader:.node=empty --loader:.png=empty ${externalFlags}`, { stdio: 'inherit' });

            // 4. Run Analysis
            console.log(`  [+] Fingerprinting structural ASTs (Bootstrap Mode)...`);
            execSync(`node src/analyze.js "${bundlePath}" --version "bootstrap/${libSafeName}" --is-bootstrap`, { stdio: 'inherit' });

            // 5. Relocate the simplified_asts.json
            const resultPath = path.join(ANALYSIS_OUTPUT, 'bootstrap', libSafeName, 'metadata', 'simplified_asts.json');

            if (fs.existsSync(resultPath)) {
                const targetStore = path.join(BOOTSTRAP_DIR, `${libSafeName}_gold_asts.json`);
                fs.copyFileSync(resultPath, targetStore);
                console.log(`  [OK] Saved ${lib.name} logic fingerprints to ${path.basename(targetStore)}`);
            }

        } catch (err) {
            console.error(`  [!] Failed to bootstrap ${lib.name}: ${err.message}`);
        }
    }

    console.log(`\n[COMPLETE] Gold Standard Library DB created in ${BOOTSTRAP_DIR}`);
}

bootstrap();
