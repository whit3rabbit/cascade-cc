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
const LIBS = JSON.parse(fs.readFileSync(path.join(__dirname, 'libs.json'), 'utf8'));


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
