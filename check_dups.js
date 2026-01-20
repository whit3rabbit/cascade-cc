const fs = require('fs');
const graphMap = JSON.parse(fs.readFileSync('cascade_graph_analysis/2.1.12/metadata/graph_map.json', 'utf8'));
const chunks = Array.isArray(graphMap) ? graphMap : graphMap.chunks;

const entries = chunks.filter(c => c.name.includes('chunk1111') || c.file?.includes('chunk1111'));
console.log(JSON.stringify(entries, null, 2));

const names = chunks.map(c => c.name);
const uniqueNames = new Set(names);
if (names.length !== uniqueNames.size) {
    console.log(`Duplicate names detected: ${names.length} total, ${uniqueNames.size} unique.`);
    // Find duplicates
    const counts = {};
    names.forEach(x => counts[x] = (counts[x] || 0) + 1);
    const dups = Object.keys(counts).filter(k => counts[k] > 1);
    console.log('Duplicates:', dups);
} else {
    console.log('No duplicate names found.');
}
