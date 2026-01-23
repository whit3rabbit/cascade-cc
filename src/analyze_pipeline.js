const { spawnSync } = require('child_process');

const args = process.argv.slice(2);
const steps = [
    'src/analyze.js',
    'src/anchor_logic.js',
    'src/classify_logic.js',
    'src/rename_chunks.js'
];

for (const script of steps) {
    const result = spawnSync('node', ['--max-old-space-size=8192', script, ...args], {
        stdio: 'inherit'
    });
    if (result.status !== 0) {
        process.exit(result.status ?? 1);
    }
}
