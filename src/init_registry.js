const fs = require('fs');
const path = require('path');

function initRegistry(version) {
    const baseDir = './cascade_graph_analysis';
    const versionPath = path.resolve(baseDir, version);
    const logicDbPath = path.join(versionPath, 'metadata', 'logic_db.json');
    const mappingPath = path.join(versionPath, 'metadata', 'mapping.json');

    if (!fs.existsSync(logicDbPath) || !fs.existsSync(mappingPath)) {
        console.error(`Error: Missing metadata for version ${version}`);
        return;
    }

    const logicDb = JSON.parse(fs.readFileSync(logicDbPath, 'utf8'));
    const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));

    // We want to create a registry that maps logic vectors (from logicDb)
    // to resolved names (from mapping).
    const registry = {};

    for (const chunk of logicDb) {
        // Find deobfuscated names associated with this chunk
        const variables = {};
        for (const [mangled, entry] of Object.entries(mapping.variables)) {
            const match = Array.isArray(entry)
                ? entry.find(e => e.source === chunk.name)
                : (entry.source === chunk.name ? entry : null);
            if (match) variables[mangled] = match.name;
        }

        const properties = {};
        for (const [mangled, entry] of Object.entries(mapping.properties)) {
            const match = Array.isArray(entry)
                ? entry.find(e => e.source === chunk.name)
                : (entry.source === chunk.name ? entry : null);
            if (match) properties[mangled] = match.name;
        }

        if (Object.keys(variables).length > 0 || Object.keys(properties).length > 0) {
            // This chunk has "Knowledge". Let's index it.
            // In a real system, we'd use a unique structural ID or category.
            // For now, we'll just store the vector and the symbol mapping.
            registry[chunk.name] = {
                vector: chunk.vector,
                symbols: chunk.symbols,
                resolved_variables: variables,
                resolved_properties: properties
            };
        }
    }

    const registryPath = path.join(baseDir, 'logic_registry.json');
    fs.writeFileSync(registryPath, JSON.stringify(registry, null, 2));
    console.log(`[+] Initialized Logic Registry at ${registryPath} with ${Object.keys(registry).length} entries.`);
}

if (require.main === module) {
    const version = process.argv[2] || '2.1.6';
    initRegistry(version);
}
