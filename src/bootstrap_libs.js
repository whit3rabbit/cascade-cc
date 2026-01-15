const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Updated Library Manifest (2026 Stable Versions)
 * Expanded to include all critical dependencies for deobfuscation mapping.
 */
const LIBS = [
    // --- The "Brain" and UI Framework ---
    { name: 'zod', version: '4.3.5' },
    { name: 'zod', version: '3.23.8' }, // Version Drift
    { name: 'react', version: '19.2.3' },
    { name: 'react', version: '18.3.1' }, // Version Drift
    { name: 'ink', version: '6.6.0' },
    { name: '@inkjs/ui', version: '2.0.0' },
    { name: 'ink-text-input', version: '6.0.0' },
    { name: 'ink-select-input', version: '6.2.0' },

    // --- Primary SDKs (Anthropic & MCP) ---
    { name: '@anthropic-ai/sdk', version: '0.71.2' },
    { name: '@anthropic-ai/sdk', version: '0.40.0' }, // Version Drift
    { name: '@modelcontextprotocol/sdk', version: '1.25.2' },

    // --- Common Utilities (High Drift) ---
    { name: 'axios', version: '1.7.9' },
    { name: 'axios', version: '0.27.2' }, // Version Drift
    { name: 'lodash-es', version: '4.17.21' },

    // ... other core libs
    { name: 'execa', version: '9.6.1' },
    { name: 'chalk', version: '5.6.2' },
    { name: 'commander', version: '14.0.2' },
    { name: 'ws', version: '8.18.3' }
];

const BOOTSTRAP_DIR = path.resolve('./ml/bootstrap_data');
const ANALYSIS_OUTPUT = path.resolve('./cascade_graph_analysis');

async function bootstrap() {
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
            execSync(`npm install ${lib.name}@${lib.version} --no-save --prefix "${libWorkDir}" --legacy-peer-deps`, { stdio: 'ignore' });

            // 2. Create Entry Point (mimic bundling)
            const entryFile = path.join(libWorkDir, 'entry.js');
            fs.writeFileSync(entryFile, `
                import * as Lib from '${lib.name}';
                console.log(Lib);
            `);

            // 3. Bundle with ESBUILD
            const bundlePath = path.join(libWorkDir, 'bundled.js');
            console.log(`  [+] Bundling with esbuild...`);

            const nodePath = path.join(libWorkDir, 'node_modules');
            execSync(`NODE_PATH="${nodePath}" npx esbuild "${entryFile}" --bundle --minify-whitespace --minify-syntax --platform=node --format=esm --target=node18 --outfile="${bundlePath}" --external:sharp --external:tree-sitter --external:tree-sitter-typescript --external:react-devtools-core --external:fsevents`, { stdio: 'inherit' });

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