#!/usr/bin/env node
require('dotenv').config();

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const DEFAULT_CONFIG_PATH = path.join('scripts', 'sweeps', 'vendor_founder_2.1.19.json');

const getArgValue = (flag) => {
    const idx = process.argv.indexOf(flag);
    if (idx === -1) return null;
    return process.argv[idx + 1] || null;
};

const toSafeName = (value) => {
    return String(value || '')
        .toLowerCase()
        .replace(/[^a-z0-9-_]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'run';
};

const ensureUniqueDir = (baseDir) => {
    if (!fs.existsSync(baseDir)) return baseDir;
    let i = 2;
    let candidate = `${baseDir}-${i}`;
    while (fs.existsSync(candidate)) {
        i += 1;
        candidate = `${baseDir}-${i}`;
    }
    return candidate;
};

const runStep = (label, cmd, args, env) => {
    console.log(`[*] ${label}: ${cmd} ${args.join(' ')}`);
    const result = spawnSync(cmd, args, {
        stdio: 'inherit',
        env
    });
    if (result.status !== 0) {
        throw new Error(`${label} failed with exit code ${result.status}`);
    }
};

const formatEnvOverrides = (env) => {
    const entries = Object.entries(env || {});
    if (entries.length === 0) return '(none)';
    return entries.map(([key, value]) => `${key}=${value}`).join(', ');
};

const loadConfig = (configPath) => {
    if (!fs.existsSync(configPath)) {
        throw new Error(`Config not found: ${configPath}`);
    }
    return JSON.parse(fs.readFileSync(configPath, 'utf8'));
};

const summarizeCounts = (graphData) => {
    const counts = {
        vendor: 0,
        founder: 0,
        family: 0,
        unknown: 0,
        total: 0
    };

    (graphData.chunks || []).forEach((chunk) => {
        counts.total += 1;
        if (chunk.category === 'vendor') counts.vendor += 1;
        else if (chunk.category === 'founder') counts.founder += 1;
        else if (chunk.category === 'family') counts.family += 1;
        else counts.unknown += 1;
    });

    const founderTotal = counts.founder + counts.family;
    const vendorPct = counts.total ? (counts.vendor / counts.total) * 100 : 0;
    const founderPct = counts.total ? (founderTotal / counts.total) * 100 : 0;

    return {
        counts,
        founderTotal,
        vendorPct: Number(vendorPct.toFixed(2)),
        founderPct: Number(founderPct.toFixed(2))
    };
};

const configPath = getArgValue('--config') || DEFAULT_CONFIG_PATH;
const config = loadConfig(configPath);

const version = getArgValue('--version') || config.version;
if (!version) {
    console.error('[!] Missing target version in config or --version');
    process.exit(1);
}

const outputRoot = getArgValue('--output-dir') || config.outputDir || path.join('cascade_graph_analysis', 'sweeps', `vendor_founder_${version}`);
const baseEnv = config.baseEnv || {};
const referenceVersion = config.referenceVersion || null;

const runs = Array.isArray(config.runs) ? config.runs : [];
if (runs.length === 0) {
    console.error('[!] Config has no runs defined.');
    process.exit(1);
}

const results = [];
const startedAt = new Date().toISOString();

for (const run of runs) {
    const runName = toSafeName(run.name || 'run');
    const runEnv = { ...process.env, ...baseEnv, ...(run.env || {}) };
    console.log(`\n[SWEEP] ${runName}`);
    console.log(`    Env: ${formatEnvOverrides(run.env || {})}`);

    runStep('Analyze', 'node', ['--max-old-space-size=8192', 'src/analyze.js', '--version', version], runEnv);
    runStep(
        'Anchor',
        'node',
        ['--max-old-space-size=8192', 'src/anchor_logic.js', version, ...(referenceVersion ? [referenceVersion] : [])],
        runEnv
    );
    runStep('Classify', 'node', ['src/classify_logic.js', version], runEnv);

    const graphPath = path.join('cascade_graph_analysis', version, 'metadata', 'graph_map.json');
    if (!fs.existsSync(graphPath)) {
        throw new Error(`Missing graph_map.json at ${graphPath}`);
    }

    const graphData = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
    const summary = summarizeCounts(graphData);

    const runDir = ensureUniqueDir(path.join(outputRoot, runName));
    fs.mkdirSync(runDir, { recursive: true });
    fs.cpSync(path.join('cascade_graph_analysis', version), runDir, { recursive: true });

    results.push({
        name: run.name || runName,
        safeName: runName,
        env: run.env || {},
        summary,
        outputDir: runDir
    });
}

fs.mkdirSync(outputRoot, { recursive: true });

const report = {
    version,
    referenceVersion,
    outputRoot,
    startedAt,
    configPath,
    baseEnv,
    runs: results
};

const reportJsonPath = path.join(outputRoot, 'report.json');
fs.writeFileSync(reportJsonPath, JSON.stringify(report, null, 2));

const reportMdPath = path.join(outputRoot, 'report.md');
const mdLines = [];
mdLines.push(`# Vendor vs Founder Sweep (${version})`);
mdLines.push('');
mdLines.push(`Started: ${startedAt}`);
mdLines.push('');
mdLines.push('Founder total = founder + family');
mdLines.push('');
mdLines.push('| Run | Env Overrides | Vendor | Founder | Family | Unknown | Total | Vendor % | Founder % |');
mdLines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- |');

results.forEach((entry) => {
    const { counts, founderTotal, vendorPct, founderPct } = entry.summary;
    mdLines.push(
        `| ${entry.safeName} | ${formatEnvOverrides(entry.env)} | ${counts.vendor} | ${founderTotal} | ${counts.family} | ${counts.unknown} | ${counts.total} | ${vendorPct} | ${founderPct} |`
    );
});

mdLines.push('');
mdLines.push(`Report JSON: ${reportJsonPath}`);
fs.writeFileSync(reportMdPath, mdLines.join('\n'));

console.log(`\n[COMPLETE] Report written to ${reportMdPath}`);
