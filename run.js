const { execSync, spawn, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const semver = require('semver');

const findPython = () => {
    const isWin = process.platform === 'win32';
    const venvPaths = [
        path.join(__dirname, '.venv'),
        path.join(__dirname, 'ml/.venv'),
        path.join(__dirname, 'ml/venv')
    ];

    const binPaths = [];
    for (const venv of venvPaths) {
        if (isWin) {
            binPaths.push(path.join(venv, 'Scripts/python.exe'));
        } else {
            binPaths.push(path.join(venv, 'bin/python3'));
            binPaths.push(path.join(venv, 'bin/python'));
        }
    }

    // Add global fallbacks
    binPaths.push('python3');
    binPaths.push('python');

    if (isWin) {
        // Additional Windows common locations or commands
        binPaths.push('py -3');
        binPaths.push('py');
    }

    for (const p of binPaths) {
        try {
            if (p.includes(path.sep) && fs.existsSync(p)) {
                return p;
            }
            // For global commands, check if they exist
            if (!p.includes(path.sep)) {
                const checkCmd = isWin ? `where ${p}` : `which ${p}`;
                execSync(checkCmd, { stdio: 'ignore' });
                return p;
            }
        } catch (e) { }
    }
    return isWin ? 'python' : 'python3';
};

const PYTHON_BIN = findPython();

const command = process.argv[2];
const args = process.argv.slice(3).map(arg => arg.replace(/[^a-zA-Z0-9.\-_=:/]/g, ''));

const VALID_COMMANDS = ['analyze', 'visualize', 'deobfuscate', 'assemble', 'anchor', 'anchor-classify', 'classify', 'propagate-names', 'train', 'bootstrap', 'clean', 'refine', 'full', 'extract-bun'];

const scripts = {
    'analyze': {
        cmd: 'node',
        args: ['--max-old-space-size=8192', 'src/analyze_pipeline.js', ...args],
        desc: 'Run CASCADE analysis'
    },
    'visualize': {
        cmd: 'npx',
        args: ['serve', '.', ...args],
        desc: 'Start visualization server'
    },
    'clean': {
        cmd: 'rm',
        args: ['-rf', 'cascade_graph_analysis', 'claude-analysis'],
        desc: 'Clean analysis outputs'
    },
    'deobfuscate': {
        cmd: 'node',
        args: ['src/deobfuscate_pipeline.js', ...args],
        desc: 'Run LLM-based deobfuscation (Stage 1 & 2). Use --rename-only or -r to just re-run renaming phase.'
    },
    'assemble': {
        cmd: 'node',
        args: ['src/assemble_final.js', ...args],
        desc: 'Assemble deobfuscated chunks into a final file structure'
    },
    'anchor': {
        cmd: 'node',
        args: ['src/anchor_logic.js', ...args],
        desc: 'Run anchoring logic to compare two versions'
    },
    'classify': {
        cmd: 'node',
        args: ['src/classify_logic.js', ...args],
        desc: 'Assign roles and propose folder structures for chunks'
    },
    'propagate-names': {
        cmd: 'node',
        args: ['src/propagate_names.js', ...args],
        desc: 'Propagate high-confidence naming hints to neighbor chunks'
    },
    'anchor-classify': {
        cmd: 'node',
        args: ['src/anchor_logic.js', ...args],
        desc: 'Run anchoring, then classify roles and propose folder structures'
    },
    'train': {
        cmd: 'node',
        args: [],
        desc: 'Initiate Model Training'
    },
    'bootstrap': {
        cmd: 'node',
        args: ['src/bootstrap_libs.js', ...args],
        desc: 'Bootstrap library DNA (Gold Standards)'
    },
    'refine': {
        cmd: 'node',
        args: ['src/refine_codebase.js', ...args],
        desc: 'Perform final LLM refinement on the assembled codebase'
    },
    'full': {
        cmd: 'node',
        args: [],
        desc: 'Run analyze -> anchor -> deobfuscate -> assemble -> refine for a version'
    },
    'extract-bun': {
        cmd: PYTHON_BIN,
        args: ['scripts/extract_bun_bundle.py', ...args],
        desc: 'Extract Bun-bundled entrypoints into cascade_graph_analysis/<version>/bun_extracted'
    }
};

const getLatestVersion = (outputRoot) => {
    if (!fs.existsSync(outputRoot)) return null;
    const versions = fs.readdirSync(outputRoot).filter(f => {
        const fullPath = path.join(outputRoot, f);
        return fs.statSync(fullPath).isDirectory() && semver.valid(f);
    });
    if (versions.length === 0) return null;
    return versions.sort(semver.rcompare)[0];
};

if (!command || !scripts[command]) {
    console.log('Usage: node run <command> [args]');
    console.log('\nAvailable commands:');
    Object.entries(scripts).forEach(([name, cfg]) => {
        console.log(`  ${name.padEnd(12)} - ${cfg.desc}`);
    });
    console.log('\nTrain args (pass after --):');
    console.log('  --device <cuda|mps|cpu|auto>');
    console.log('  --max_nodes <int>');
    console.log('  --batch_size <int>');
    console.log('  --lr <float>');
    console.log('  --margin <float>');
    console.log('  --embed_dim <int>');
    console.log('  --hidden_dim <int>');
    console.log('  --epochs <int>');
    console.log('  --preset <name>');
    console.log('  --checkpoint_interval <int>');
    console.log('  --lr_decay_epoch <int>');
    console.log('  --lr_decay_factor <float>');
    console.log('  --sweep');
    console.log('  --finetune');
    console.log('  --val_library <name>[,<name>...] (repeatable)');
    console.log('  --val_lib_count <int>');
    console.log('  --val_split <float>');
    console.log('  --val_max_chunks <int>');
    process.exit(1);
}

const config = scripts[command];
console.log(`[*] Running: ${config.cmd} ${config.args.join(' ')}`);

// The original code used a generic spawn for all commands.
// The instruction implies a switch statement for specific commands like 'train'.
// We will adapt the execution flow to use a switch for commands that require specific handling
// and fall back to the generic spawn for others.

switch (command) {
        case 'analyze':
        case 'visualize':
        case 'clean':
        case 'deobfuscate':
        case 'assemble':
        case 'refine':
        case 'anchor': {
            const result = spawnSync(config.cmd, config.args, {
                stdio: 'inherit',
                shell: true,
                env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
            });
    
            if (result.status !== 0) {
                process.exit(result.status);
            }
            break;
        }

    case 'classify':
    case 'propagate-names': {
        const child = spawn(config.cmd, config.args, {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
        });
        child.on('exit', (code) => {
            process.exit(code);
        });
        break;
    }

    case 'extract-bun': {
        const child = spawn(config.cmd, config.args, {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
        });
        child.on('exit', (code) => {
            process.exit(code);
        });
        break;
    }

    case 'anchor-classify': {
        const extractTargetVersion = (argv) => {
            const versionIdx = argv.indexOf('--version');
            const nonFlagArgs = [];
            for (let i = 0; i < argv.length; i++) {
                const arg = argv[i];
                if (arg === '--version') {
                    i += 1;
                    continue;
                }
                if (arg.startsWith('--')) continue;
                nonFlagArgs.push(arg);
            }

            let targetVersion = versionIdx !== -1 ? argv[versionIdx + 1] : nonFlagArgs[0];
            if (targetVersion) return targetVersion;

            const baseDir = './cascade_graph_analysis';
            if (!fs.existsSync(baseDir)) return null;
            const dirs = fs.readdirSync(baseDir).filter(d => {
                return fs.statSync(path.join(baseDir, d)).isDirectory() && d !== 'bootstrap';
            }).sort().reverse();
            return dirs.length > 0 ? dirs[0] : null;
        };

        const targetVersion = extractTargetVersion(args);
        if (!targetVersion) {
            console.log('Usage: node run anchor-classify <target_version> [reference_version]');
            process.exit(1);
        }

        const anchorChild = spawn('node', ['src/anchor_logic.js', ...args], {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
        });
            anchorChild.on('exit', (code) => {
                if (code !== 0) process.exit(code);
                            const classifyChild = spawn('node', ['src/classify_logic.js', targetVersion], {
                                stdio: 'inherit',
                                shell: true,
                                env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
                            });                classifyChild.on('exit', (classifyCode) => {
                    if (classifyCode !== 0) process.exit(classifyCode);
                    const propagateChild = spawn('node', ['src/propagate_names.js', targetVersion], {
                        stdio: 'inherit',
                        shell: true,
                        env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
                    });
                    propagateChild.on('exit', (propCode) => {
                        process.exit(propCode);
                    });
                });
            });
        break;
    }

    case 'full': {
        const extractTargetVersion = (argv) => {
            const versionIdx = argv.indexOf('--version');
            const nonFlagArgs = [];
            for (let i = 0; i < argv.length; i++) {
                const arg = argv[i];
                if (arg === '--version') {
                    i += 1;
                    continue;
                }
                if (arg.startsWith('--')) continue;
                nonFlagArgs.push(arg);
            }
            const targetVersion = versionIdx !== -1 ? argv[versionIdx + 1] : nonFlagArgs[0];
            const referenceVersion = nonFlagArgs[1] || null;
            return { targetVersion, referenceVersion };
        };

        const { targetVersion, referenceVersion } = extractTargetVersion(args);

        const runStep = (label, cmd, stepArgs) => {
            console.log(`[*] Running (${label}): ${cmd} ${stepArgs.join(' ')}`);
            const result = spawnSync(cmd, stepArgs, {
                stdio: 'inherit',
                shell: true,
                env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
            });
            const code = typeof result.status === 'number' ? result.status : 1;
            if (code !== 0) process.exit(code);
        };

        let resolvedVersion = targetVersion || null;
        const analysisArgs = ['--max-old-space-size=8192', 'src/analyze_pipeline.js'];
        if (resolvedVersion) analysisArgs.push('--version', resolvedVersion);
        runStep('analyze', 'node', analysisArgs);

        if (!resolvedVersion) {
            resolvedVersion = getLatestVersion('./cascade_graph_analysis');
            if (!resolvedVersion) {
                console.log('[!] Could not determine latest version after analyze.');
                process.exit(1);
            }
            console.log(`[*] No version specified. Using latest: ${resolvedVersion}`);
        }

        const deobfuscateArgs = ['--max-old-space-size=8192', 'src/deobfuscate_pipeline.js', resolvedVersion];
        if (!args.includes('--skip-vendor')) {
            deobfuscateArgs.push('--skip-vendor');
        }
        runStep('deobfuscate', 'node', deobfuscateArgs);
        runStep('assemble', 'node', ['--max-old-space-size=8192', 'src/assemble_final.js', resolvedVersion]);
        runStep('refine', 'node', ['src/refine_codebase.js', resolvedVersion]);
        process.exit(0);
        break;
    }

    case 'bootstrap': {
        console.log(`[*] Initiating Library Bootstrapping...`);
        const runPython = (scriptPath, args = []) => {
            return spawnSync(PYTHON_BIN, [scriptPath, ...args], { stdio: 'inherit', shell: true });
        };

        const bootstrapArgs = [...args];
        if (process.env.CI === 'true' && !bootstrapArgs.includes('--yes')) {
            bootstrapArgs.push('--yes');
        }
        spawnSync('node', ['src/ingest_custom_gold.js'], { stdio: 'inherit' });
        spawnSync('node', ['src/bootstrap_custom_gold.js', ...bootstrapArgs], { stdio: 'inherit' });
        spawnSync('node', ['src/bootstrap_libs.js', ...bootstrapArgs], { stdio: 'inherit' });

        console.log(`[*] Vectorizing Bootstrap Data...`);
        const bootstrapDir = './cascade_graph_analysis/bootstrap';

        if (fs.existsSync(bootstrapDir)) {
            const libs = fs.readdirSync(bootstrapDir).filter(f => fs.statSync(path.join(bootstrapDir, f)).isDirectory());
            for (const lib of libs) {
                console.log(`    [+] Vectorizing ${lib}...`);
                spawnSync(PYTHON_BIN, ['ml/vectorize.py', path.join(bootstrapDir, lib)], { stdio: 'inherit' });
            }
        }

        console.log(`[*] Updating Logic Registry...`);
        spawnSync('node', ['src/update_registry_from_bootstrap.js'], { stdio: 'inherit' });

        process.exit(0);
        break;
    }

    case 'train': {
        const bootstrapDir = './ml/bootstrap_data';

        console.log(`[*] Initiating Model Training on Gold Standards...`);
        if (args.includes('--help') || args.includes('-h')) {
            spawnSync(PYTHON_BIN, ['ml/train.py', '--help'], { stdio: 'inherit', shell: true });
            process.exit(0);
        }
        const child = spawn(PYTHON_BIN, ['ml/train.py', bootstrapDir, ...args], {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
        });
        child.on('exit', (code) => {
            process.exit(code);
        });
        break;
    }

    default: {
        const child = spawn(config.cmd, config.args, {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, PYTHON_BIN: PYTHON_BIN, GEMINI_TEMP_DIR: process.env.GEMINI_TEMP_DIR }
        });
        child.on('exit', (code) => {
            process.exit(code);
        });
        break;
    }
}


if (command === 'visualize') {
    const url = 'http://localhost:3000/visualizer/';
    const openCmd = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'start' : 'xdg-open';

    setTimeout(() => {
        try {
            console.log(`[*] Opening browser to: ${url}`);
            execSync(`${openCmd} ${url}`);
        } catch (e) {
            console.warn(`[!] Failed to open browser automatically. Please visit ${url} manually.`);
        }
    }, 2000);
}
