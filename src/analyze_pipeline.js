const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const baseDir = './cascade_graph_analysis';

const parseArgs = (argv) => {
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
  const referenceVersion = versionIdx !== -1 ? nonFlagArgs[0] : nonFlagArgs[1];
  return { targetVersion, referenceVersion };
};

const detectLatestVersion = () => {
  if (!fs.existsSync(baseDir)) return null;
  const dirs = fs.readdirSync(baseDir)
    .filter((d) => {
      if (d === 'bootstrap' || d === 'sweeps') return false;
      const fullPath = path.join(baseDir, d);
      if (!fs.statSync(fullPath).isDirectory()) return false;
      return /^\\d+\\.\\d+\\.\\d+/.test(d) || d.startsWith('custom-');
    })
    .sort()
    .reverse();
  return dirs.length > 0 ? dirs[0] : null;
};

const runStep = (script, stepArgs) => {
  const result = spawnSync('node', ['--max-old-space-size=8192', script, ...stepArgs], {
    stdio: 'inherit',
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
};

const { targetVersion: initialTarget, referenceVersion } = parseArgs(args);

// Ensure GEMINI_TEMP_DIR is passed down as a command-line argument for analyze.js
const geminiTempDir = process.env.GEMINI_TEMP_DIR || '/Users/whit3rabbit/.gemini/tmp/00150962b70d3731010c9badd4003d7d7023e6e06d18dde520c44bbc52c5c1d0'; // Fallback for testing
runStep('src/analyze.js', [...args, `--gemini-temp-dir=${geminiTempDir}`]);

let resolvedTarget = initialTarget;
if (!resolvedTarget) {
  resolvedTarget = detectLatestVersion();
}
if (!resolvedTarget) {
  console.error('[!] Failed to resolve target version after analyze.');
  process.exit(1);
}

runStep('src/anchor_logic.js', [resolvedTarget, ...(referenceVersion ? [referenceVersion] : [])]);
runStep('src/classify_logic.js', [resolvedTarget]);
runStep('src/propagate_names.js', [resolvedTarget]);
runStep('src/rename_chunks.js', [resolvedTarget]);
