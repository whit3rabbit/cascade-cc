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
    const shouldPrompt = process.argv.includes('--prompt');
    const shouldForce = process.argv.includes('--force');
    const isVerbose = process.argv.includes('--verbose');
    const includeNative = process.argv.includes('--include-native');
    const useBun = process.argv.includes('--bun');
    const useEsbuild = process.argv.includes('--esbuild');
    const BUNDLERS = (useBun || useEsbuild)
        ? [
            ...(useEsbuild ? ['esbuild'] : []),
            ...(useBun ? ['bun'] : [])
        ]
        : ['esbuild', 'bun'];

    const readPackageJson = (pkgJsonPath) => {
        try {
            return JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
        } catch {
            return null;
        }
    };

    const getEntryCandidates = (pkgJson) => {
        const candidates = [];
        const exportsField = pkgJson && pkgJson.exports;
        const rootExport = exportsField && typeof exportsField === 'object'
            ? (exportsField['.'] || exportsField)
            : (typeof exportsField === 'string' ? exportsField : null);

        if (rootExport) {
            if (typeof rootExport === 'string') {
                candidates.push(rootExport);
            } else if (typeof rootExport === 'object') {
                if (typeof rootExport.import === 'string') candidates.push(rootExport.import);
                if (typeof rootExport.require === 'string') candidates.push(rootExport.require);
                if (typeof rootExport.default === 'string') candidates.push(rootExport.default);
            }
        }

        if (pkgJson && typeof pkgJson.module === 'string') candidates.push(pkgJson.module);
        if (pkgJson && typeof pkgJson.main === 'string') candidates.push(pkgJson.main);
        candidates.push('index.js');
        return candidates;
    };

    const findExistingEntry = (pkgPath, pkgJson) => {
        const candidates = getEntryCandidates(pkgJson);
        for (const candidate of candidates) {
            const normalized = candidate.replace(/^\.\//, '');
            const entryPath = path.join(pkgPath, normalized);
            if (fs.existsSync(entryPath)) return { entryPath, candidate };
        }
        return null;
    };

    const isNativeModule = (pkgJson) => {
        if (!pkgJson) return false;
        if (pkgJson.gypfile) return true;
        if (pkgJson.binary) return true;
        const files = Array.isArray(pkgJson.files) ? pkgJson.files : [];
        if (files.some(file => file.includes('prebuilds'))) return true;
        const deps = pkgJson.dependencies || {};
        if (deps['node-gyp-build'] || deps['node-addon-api']) return true;
        const scripts = pkgJson.scripts || {};
        if (typeof scripts.install === 'string' && scripts.install.includes('node-gyp')) return true;
        return false;
    };

    if (shouldPrompt) {
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
        for (const bundler of BUNDLERS) {
            const libSafeName = `${lib.name.replace(/[@\/]/g, '_')}_v${lib.version.replace(/\./g, '_')}_${bundler}`;
            const libWorkDir = path.join(BOOTSTRAP_DIR, libSafeName);
            const targetStore = path.join(BOOTSTRAP_DIR, `${libSafeName}_gold_asts.json`);
            const nodeModulesDir = path.join(libWorkDir, 'node_modules');

            console.log(`\n[${lib.name.toUpperCase()} @ ${lib.version} | ${bundler}]`);
            if (!fs.existsSync(libWorkDir)) fs.mkdirSync(libWorkDir, { recursive: true });

            try {
                if (fs.existsSync(targetStore) && !shouldForce) {
                    if (isVerbose) {
                        console.log(`  [SKIP] Found existing fingerprints (${path.basename(targetStore)}).`);
                    }
                    continue;
                }

                // 1. Isolated Install for this specific version
                const pkgPath = path.join(nodeModulesDir, ...lib.name.split('/'));
                const pkgJsonPath = path.join(pkgPath, 'package.json');
                let hasInstalled = fs.existsSync(pkgJsonPath);
                let matchesVersion = false;
                let existingEntry = null;
                if (hasInstalled) {
                    try {
                        const pkgJson = readPackageJson(pkgJsonPath);
                        const installedVersion = pkgJson && pkgJson.version;
                        matchesVersion = lib.version === 'latest' || installedVersion === lib.version;
                        existingEntry = pkgJson ? findExistingEntry(pkgPath, pkgJson) : null;
                    } catch {
                        matchesVersion = false;
                    }
                }

                if (shouldForce || !hasInstalled || !matchesVersion || !existingEntry) {
                    console.log(`  [+] Installing ${lib.name}@${lib.version}...`);
                    const installCmd = `npm install ${lib.name}@${lib.version} ${lib.peerDeps ? lib.peerDeps.join(' ') : ''} --no-save --prefix "${libWorkDir}" --legacy-peer-deps --install-links`;
                    execSync(installCmd, { stdio: 'ignore' });
                    const pkgJson = readPackageJson(pkgJsonPath);
                    existingEntry = pkgJson ? findExistingEntry(pkgPath, pkgJson) : null;
                    if (!existingEntry) {
                        console.warn(`  [WARN] Missing entry files after install for ${lib.name}@${lib.version}. Skipping.`);
                        continue;
                    }
                } else if (isVerbose) {
                    console.log(`  [SKIP] Existing node_modules match ${lib.name}@${lib.version}.`);
                }

                // 2. Create Entry Point (mimic bundling)
                const pkgJson = readPackageJson(pkgJsonPath);
                if (isNativeModule(pkgJson) && !includeNative) {
                    console.warn(`  [WARN] Native module detected for ${lib.name}@${lib.version}. Skipping bundle.`);
                    continue;
                }
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

                // 3. Bundle
                const bundlePath = path.join(libWorkDir, 'bundled.js');
                if (bundler === 'bun') {
                    console.log(`  [+] Bundling with Bun...`);
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

                    const externalFlags = externals.map(ext => `-e ${ext}`).join(' ');
                    const bunCmd = `bun build "${entryFile}" --minify --target node --outfile "${bundlePath}" ${externalFlags}`;
                    execSync(bunCmd, { stdio: 'inherit' });
                } else {
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

                    const hasMissingEsmEntry = () => {
                        const pkgJson = readPackageJson(pkgJsonPath);
                        const exportsField = pkgJson && pkgJson.exports;
                        const rootExport = exportsField && typeof exportsField === 'object'
                            ? (exportsField['.'] || exportsField)
                            : (typeof exportsField === 'string' ? exportsField : null);
                        const importTarget = rootExport && typeof rootExport === 'object' && typeof rootExport.import === 'string'
                            ? rootExport.import
                            : null;
                        if (!importTarget) return false;
                        const candidate = path.join(pkgPath, importTarget.replace(/^\.\//, ''));
                        return !fs.existsSync(candidate);
                    };

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
                    const nodeNativeExternal = '--external:*.node';
                    const useCjsConditions = hasMissingEsmEntry();
                    const conditionsFlags = useCjsConditions ? '--conditions=default,require --main-fields=main' : '';
                    execSync(`NODE_PATH="${nodePath}" npx -y esbuild "${entryFile}" --bundle --minify-whitespace --minify-syntax --platform=node --format=esm --target=node18 --outfile="${bundlePath}" --loader:.node=empty --loader:.png=empty ${nodeNativeExternal} ${conditionsFlags} ${externalFlags}`, { stdio: 'inherit' });
                }

                // 4. Run Analysis
                console.log(`  [+] Fingerprinting structural ASTs (Bootstrap Mode)...`);
                execSync(`node src/analyze.js "${bundlePath}" --version "bootstrap/${libSafeName}" --is-bootstrap`, { stdio: 'inherit' });

                // 5. Relocate the simplified_asts.json
                const resultPath = path.join(ANALYSIS_OUTPUT, 'bootstrap', libSafeName, 'metadata', 'simplified_asts.json');

                if (fs.existsSync(resultPath)) {
                    fs.copyFileSync(resultPath, targetStore);
                    console.log(`  [OK] Saved ${lib.name} logic fingerprints to ${path.basename(targetStore)}`);
                }

            } catch (err) {
                console.error(`  [!] Failed to bootstrap ${lib.name} (${bundler}): ${err.message}`);
            }
        }
    }

    console.log(`\n[COMPLETE] Gold Standard Library DB created in ${BOOTSTRAP_DIR}`);
}

bootstrap();
