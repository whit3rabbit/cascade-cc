const fs = require('fs');
const path = require('path');

const targets = [
    'src/analyze_pipeline.js',
    'src/classify_logic.js',
    'src/rename_chunks.js',
    'src/analyze.js'
];

const badLiteral = '/^\\\\d+\\\\.\\\\d+\\\\.\\\\d+/';
let hasError = false;

for (const relPath of targets) {
    const fullPath = path.resolve(relPath);
    if (!fs.existsSync(fullPath)) {
        console.warn(`[WARN] Missing file: ${relPath}`);
        continue;
    }
    const contents = fs.readFileSync(fullPath, 'utf8');
    if (contents.includes(badLiteral)) {
        console.error(`[FAIL] Escaped semver regex literal found in ${relPath}: ${badLiteral}`);
        hasError = true;
    }
}

const semverRe = /^\d+\.\d+\.\d+/;
if (!semverRe.test('2.1.29')) {
    console.error('[FAIL] Semver regex sanity check failed on "2.1.29".');
    hasError = true;
}
if (semverRe.test('custom-abc')) {
    console.error('[FAIL] Semver regex sanity check incorrectly matched "custom-abc".');
    hasError = true;
}

if (hasError) {
    process.exit(1);
}

console.log('[OK] Version detection regex sanity checks passed.');
