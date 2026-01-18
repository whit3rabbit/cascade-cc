const fs = require('fs');
const path = require('path');

const VERSION = '2.1.12';
const GRAPH_MAP_PATH = `./cascade_graph_analysis/${VERSION}/metadata/graph_map.json`;

function findRoots() {
    const data = JSON.parse(fs.readFileSync(GRAPH_MAP_PATH, 'utf8'));
    const chunks = data.chunks;

    const inDegree = {};
    chunks.forEach(c => inDegree[c.name] = 0);

    chunks.forEach(c => {
        c.outbound.forEach(neighbor => {
            if (inDegree[neighbor] !== undefined) {
                inDegree[neighbor]++;
            }
        });
    });

    const roots = chunks.map(c => ({
        name: c.name,
        in: inDegree[c.name],
        out: c.outbound.length,
        size: c.tokens,
        centrality: r = c.centrality,
        suggested: c.suggestedFilename
    })).filter(c => c.in === 0 || (c.in < 3 && c.out > 10));

    roots.sort((a, b) => b.out - a.out);
    console.log(JSON.stringify(roots.slice(0, 20), null, 2));
}

findRoots();
