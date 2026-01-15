const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Updated Library Manifest (2026 Stable Versions)
 * Expanded to include all critical dependencies for deobfuscation mapping.
 */
const LIBS = [
    // --- The "Brain" and UI Framework ---
    { name: 'zod', version: '4.3.5' }, // Matching your Zod 4+ target
    { name: 'react', version: '19.2.3' },
    { name: 'ink', version: '6.6.0' },
    { name: '@inkjs/ui', version: '2.0.0' },
    { name: 'ink-text-input', version: '6.0.0' },
    { name: 'ink-select-input', version: '6.2.0' },

    // --- Primary SDKs (Anthropic & MCP) ---
    { name: '@anthropic-ai/sdk', version: '0.71.2' },
    { name: '@modelcontextprotocol/sdk', version: '1.25.2' },
    { name: '@anthropic-ai/bedrock-sdk', version: '0.26.0' },
    { name: '@anthropic-ai/vertex-sdk', version: '0.14.0' },

    // --- Infrastructure & Cloud (AWS) ---
    { name: '@aws-sdk/client-bedrock-runtime', version: '3.962.0' },
    { name: '@aws-sdk/client-s3', version: '3.958.0' },
    { name: '@aws-sdk/credential-providers', version: '3.958.0' },
    { name: 'axios', version: '1.7.9' },
    { name: 'statsig-js', version: '5.1.0' },

    // --- Parsers & AST Tools (Crucial for Deobfuscation Mapping) ---
    { name: 'tree-sitter', version: '0.21.0' },
    { name: 'tree-sitter-typescript', version: '0.23.2' },
    { name: 'web-tree-sitter', version: '0.26.3' },
    { name: 'jsdom', version: '27.4.0' },
    { name: 'marked', version: '17.0.1' },
    { name: 'turndown', version: '7.2.2' },
    { name: 'ajv', version: '8.17.1' },

    // --- System & Utilities ---
    { name: 'execa', version: '9.6.1' },
    { name: 'chokidar', version: '5.0.0' },
    { name: 'lodash-es', version: '4.17.22' },
    { name: 'chalk', version: '5.6.2' },
    { name: 'commander', version: '14.0.2' },
    { name: 'fuse.js', version: '7.1.0' },
    { name: 'sharp', version: '0.34.5' },
    { name: 'ws', version: '8.18.3' },

    // --- Observability ---
    { name: '@opentelemetry/api', version: '1.9.0' },
    { name: '@sentry/node', version: '10.32.1' }
];

const BOOTSTRAP_DIR = path.resolve('./ml/bootstrap_data');
const ANALYSIS_OUTPUT = path.resolve('./cascade_graph_analysis');

async function bootstrap() {
    console.log(`[*] STARTING GOLD STANDARD BOOTSTRAP (ZOD 4+)`);

    if (!fs.existsSync(BOOTSTRAP_DIR)) fs.mkdirSync(BOOTSTRAP_DIR, { recursive: true });

    for (const lib of LIBS) {
        const libSafeName = lib.name.replace(/[@\/]/g, '_');
        const libWorkDir = path.join(BOOTSTRAP_DIR, libSafeName);

        console.log(`\n[${lib.name.toUpperCase()}]`);
        if (!fs.existsSync(libWorkDir)) fs.mkdirSync(libWorkDir, { recursive: true });

        try {
            // 1. Install specific version
            console.log(`  [+] Installing ${lib.name}@${lib.version}...`);
            // Using --no-save to prevent polluting a local package.json
            execSync(`npm install ${lib.name}@${lib.version} --no-save --prefix "${libWorkDir}"`, { stdio: 'ignore' });

            // 2. Create Entry Point (mimic bundling)
            const entryFile = path.join(libWorkDir, 'entry.js');
            fs.writeFileSync(entryFile, `
                import * as Lib from '${lib.name}';
                console.log(Lib);
            `);

            // 3. Bundle with ESBUILD
            const bundlePath = path.join(libWorkDir, 'bundled.js');
            console.log(`  [+] Bundling with esbuild (minify: true, format: esm)...`);
            // Note: We include --external for binary/heavy modules that shouldn't be flattened
            execSync(`npx esbuild "${entryFile}" --bundle --minify --platform=node --format=esm --target=node18 --outfile="${bundlePath}" --external:sharp --external:tree-sitter --external:tree-sitter-typescript --external:react-devtools-core --external:fsevents`, { stdio: 'inherit' });

            // 4. Run Analysis
            console.log(`  [+] Fingerprinting structural ASTs...`);
            execSync(`node src/analyze.js "${bundlePath}" --version "bootstrap/${libSafeName}"`, { stdio: 'inherit' });

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