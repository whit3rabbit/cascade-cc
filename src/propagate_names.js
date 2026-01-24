const fs = require('fs');
const path = require('path');

function extractLibFromLabel(label) {
    if (!label) return null;
    const idx = label.indexOf('_');
    return idx === -1 ? null : label.slice(0, idx);
}

function getLatestVersion(outputRoot) {
    if (!fs.existsSync(outputRoot)) return null;
    const dirs = fs.readdirSync(outputRoot).filter(d => {
        const full = path.join(outputRoot, d);
        return fs.statSync(full).isDirectory() && d !== 'bootstrap';
    }).sort().reverse();
    return dirs[0] || null;
}

function propagateNames(version) {
    const outputRoot = './cascade_graph_analysis';
    if (!version) {
        version = getLatestVersion(outputRoot);
        if (!version) {
            console.error(`[!] No versions found in ${outputRoot}`);
            process.exit(1);
        }
        console.log(`[*] No version specified. Defaulting to latest: ${version}`);
    }

    const versionPath = path.join(outputRoot, version);
    const mappingPath = path.join(versionPath, 'metadata', 'mapping.json');
    const graphMapPath = path.join(versionPath, 'metadata', 'graph_map.json');
    const neighborBoostPath = path.join(versionPath, 'metadata', 'neighbor_boosts.json');

    if (!fs.existsSync(mappingPath) || !fs.existsSync(graphMapPath)) {
        console.error(`[!] Missing mapping or graph map for ${version}`);
        process.exit(1);
    }

    const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
    const graphRaw = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
    const chunks = Array.isArray(graphRaw) ? graphRaw : (graphRaw.chunks || []);

    const threshold = parseFloat(process.env.PROPAGATE_NAME_THRESHOLD) || 0.98;
    const neighborHints = mapping.neighbor_hints || {};
    let updates = 0;

    if (fs.existsSync(neighborBoostPath)) {
        const boosts = JSON.parse(fs.readFileSync(neighborBoostPath, 'utf8'));
        Object.entries(boosts).forEach(([neighbor, hint]) => {
            if (!hint || !hint.lib) return;
            const existing = neighborHints[neighbor];
            if (!existing || (existing.similarity || 0) < (hint.confidence || 0)) {
                neighborHints[neighbor] = {
                    lib: hint.lib,
                    similarity: hint.confidence || 0,
                    source: hint.source || 'neighbor_boosts'
                };
                updates++;
            }
        });
    } else {
        chunks.forEach(chunk => {
            const sim = chunk.matchSimilarityBoosted ?? chunk.matchSimilarity ?? 0;
            if (sim < threshold) return;
            const lib = extractLibFromLabel(chunk.matchLabel);
            if (!lib) return;

            const neighbors = [...(chunk.neighbors || []), ...(chunk.outbound || [])];
            neighbors.forEach(neighbor => {
                const existing = neighborHints[neighbor];
                if (!existing || (existing.similarity || 0) < sim) {
                    neighborHints[neighbor] = {
                        lib,
                        similarity: sim,
                        source: chunk.name
                    };
                    updates++;
                }
            });
        });
    }

    mapping.neighbor_hints = neighborHints;
    mapping.metadata = mapping.metadata || {};
    mapping.metadata.last_propagated = new Date().toISOString();

    fs.writeFileSync(mappingPath, JSON.stringify(mapping, null, 2));
    console.log(`[+] Propagated ${updates} neighbor hints into mapping.json`);
}

if (require.main === module) {
    const version = process.argv[2];
    propagateNames(version);
}
