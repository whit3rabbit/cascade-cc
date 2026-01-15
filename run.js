const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const command = process.argv[2];
const args = process.argv.slice(3);

const VALID_COMMANDS = ['analyze', 'visualize', 'deobfuscate', 'assemble', 'anchor', 'train', 'bootstrap', 'clean'];

const scripts = {
    'analyze': {
        cmd: 'node',
        args: ['--max-old-space-size=8192', 'src/analyze.js', ...args],
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
    'train': {
        cmd: 'node',
        args: [],
        desc: 'Initiate Model Training'
    },
    'bootstrap': {
        cmd: 'node',
        args: ['src/bootstrap_libs.js', ...args],
        desc: 'Bootstrap library DNA (Gold Standards)'
    }
};

if (!command || !scripts[command]) {
    console.log('Usage: node run <command> [args]');
    console.log('\nAvailable commands:');
    const findPython = () => {
        const paths = [
            path.join(__dirname, '.venv/bin/python3'),
            path.join(__dirname, 'ml/.venv/bin/python3'),
            'python3',
            'python'
        ];
        for (const p of paths) {
            try {
                if (fs.existsSync(p)) return p;
            } catch (e) { }
        }
        return 'python3';
    };

    const PYTHON_BIN = findPython();
    Object.entries(scripts).forEach(([name, cfg]) => {
        console.log(`  ${name.padEnd(12)} - ${cfg.desc}`);
    });
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
    case 'anchor': {
        const child = spawn(config.cmd, config.args, {
            stdio: 'inherit',
            shell: true
        });
        child.on('exit', (code) => {
            process.exit(code);
        });
        break;
    }

    case 'bootstrap': {
        console.log(`[*] Initiating Library Bootstrapping...`);
        const { spawnSync } = require('child_process');
        const runPython = (scriptPath, args = []) => {
            return spawnSync(PYTHON_BIN, [scriptPath, ...args], { stdio: 'inherit', shell: true });
        };

        spawnSync('node', ['src/bootstrap_libs.js', ...args], { stdio: 'inherit' });

        console.log(`[*] Vectorizing Bootstrap Data...`);
        const bootstrapDir = './cascade_graph_analysis/bootstrap';

        if (fs.existsSync(bootstrapDir)) {
            const libs = fs.readdirSync(bootstrapDir).filter(f => fs.statSync(path.join(bootstrapDir, f)).isDirectory());
            for (const lib of libs) {
                console.log(`    [+] Vectorizing ${lib}...`);
                spawnSync(pythonEnv, ['ml/vectorize.py', path.join(bootstrapDir, lib)], { stdio: 'inherit' });
            }
        }

        console.log(`[*] Updating Logic Registry...`);
        spawnSync('node', ['src/update_registry_from_bootstrap.js'], { stdio: 'inherit' });

        process.exit(0);
        break;
    }

    case 'train': {
        const bootstrapDir = './ml/bootstrap_data';
        const pythonEnv = fs.existsSync('./ml/venv/bin/python3') ? './ml/venv/bin/python3' : 'python3';

        console.log(`[*] Initiating Model Training on Gold Standards...`);
        const child = spawn(pythonEnv, ['ml/train.py', bootstrapDir], {
            stdio: 'inherit',
            shell: true
        });
        child.on('exit', (code) => {
            process.exit(code);
        });
        break;
    }

    default: {
        const child = spawn(config.cmd, config.args, {
            stdio: 'inherit',
            shell: true
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

