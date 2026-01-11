const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const command = process.argv[2];
const args = process.argv.slice(3);

const scripts = {
    'analyze': {
        cmd: 'node',
        args: ['--max-old-space-size=8192', 'analyze.js', ...args],
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

child.on('exit', (code) => {
    process.exit(code);
});
