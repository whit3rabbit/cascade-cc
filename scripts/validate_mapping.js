const fs = require('fs');
const path = require('path');

const OUTPUT_ROOT = './cascade_graph_analysis';
const version = '2.1.12'; // Target version from the crash logs
const mappingPath = path.join(OUTPUT_ROOT, version, 'metadata', 'mapping.json');

if (!fs.existsSync(mappingPath)) {
    console.error(`Mapping file not found at ${mappingPath}`);
    process.exit(1);
}

const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
let issues = 0;

console.log('Scanning variables...');
for (const [key, val] of Object.entries(mapping.variables || {})) {
    if (!val || typeof val.name !== 'string') {
        console.log(`[VAR] ${key} is invalid:`, val);
        issues++;
    }
}

console.log('Scanning properties...');
for (const [key, val] of Object.entries(mapping.properties || {})) {
    if (!val || typeof val.name !== 'string') {
        console.log(`[PROP] ${key} is invalid:`, val);
        issues++;
    }
}

console.log(`\nFound ${issues} invalid entries.`);
