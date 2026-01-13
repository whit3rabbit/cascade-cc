const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const command = process.argv[2];
const args = process.argv.slice(3);

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
        desc: 'Run LLM-based deobfuscation (Stage 1 & 2)'
    },
    'assemble': {
        cmd: 'node',
        args: ['src/assemble_final.js', ...args],
        desc: 'Assemble deobfuscated chunks into a final file structure'
    }
};

if (!command || !scripts[command]) {
    console.log('Usage: node run <command> [args]');
    console.log('\nAvailable commands:');
    Object.entries(scripts).forEach(([name, cfg]) => {
        console.log(`  ${name.padEnd(12)} - ${cfg.desc}`);
    });
    process.exit(1);
}

const config = scripts[command];
console.log(`[*] Running: ${config.cmd} ${config.args.join(' ')}`);

const child = spawn(config.cmd, config.args, {
    stdio: 'inherit',
    shell: true
});

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

child.on('exit', (code) => {
    process.exit(code);
});
