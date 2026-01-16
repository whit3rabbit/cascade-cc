/**
 * Logic Registry Initializer for Cold Start
 * 
 * This script populates logic_registry.json using vectorized bootstrap data.
 * Since bootstrap data consists of clean, non-obfuscated libraries, 
 * every symbol is mapped to itself.
 */
const fs = require('fs');
const path = require('path');

const bootstrapDir = './cascade_graph_analysis/bootstrap';
const registryPath = './cascade_graph_analysis/logic_registry.json';

async function updateRegistryFromBootstrap() {
    const registry = fs.existsSync(registryPath)
        ? JSON.parse(fs.readFileSync(registryPath, 'utf8'))
        : {};

    if (!fs.existsSync(bootstrapDir)) {
        console.error(`Error: Bootstrap directory not found at ${bootstrapDir}`);
        process.exit(1);
    }

    const libraries = fs.readdirSync(bootstrapDir).filter(f => {
        return fs.statSync(path.join(bootstrapDir, f)).isDirectory();
    });

    console.log(`[*] Found ${libraries.length} libraries in bootstrap directory.`);

    let totalEntries = 0;

    for (const lib of libraries) {
        const libPath = path.join(bootstrapDir, lib);
        const logicDbPath = path.join(libPath, 'metadata', 'logic_db.json');

        if (!fs.existsSync(logicDbPath)) {
            // console.warn(`[!] No logic_db.json found for library ${lib}. skipping.`);
            continue;
        }

        console.log(`[*] Processing library: ${lib}`);
        const logicDb = JSON.parse(fs.readFileSync(logicDbPath, 'utf8'));

        for (const chunk of logicDb) {
            // For bootstrap data, all symbols are resolved to themselves
            const variables = {};
            const properties = {};

            // In bootstrap data, symbols are the original names.
            // We map each symbol to itself. 
            // We put them in both variables and properties to be safe, 
            // as anchor_logic.js will favor properties if present.
            for (const symbolObj of chunk.symbols) {
                const symbol = typeof symbolObj === 'string' ? symbolObj : symbolObj.name;
                const whitelist = ['_', '$', 'z', 'd', 'e', 'i', 'j', 'k']; // Standard library/loop symbols
                if (!symbol || (symbol.length < 2 && !whitelist.includes(symbol))) continue; // Skip single chars if not in whitelist

                // Skip common minified patterns (e.g., _a, t)
                if (symbol.length === 2 && symbol.startsWith('_')) continue;
                if (['if', 'for', 'let', 'var', 'const', 'try', 'catch', 'map', 'set'].includes(symbol)) continue;

                variables[symbol] = {
                    name: symbol,
                    confidence: 1.0,
                    source: 'bootstrap'
                };
                properties[symbol] = {
                    name: symbol,
                    confidence: 1.0,
                    source: 'bootstrap'
                };
            }

            // Group by library and chunk name to avoid collisions
            const registryKey = `${lib}_${chunk.name}`;
            registry[registryKey] = {
                vector: chunk.vector,
                symbols: chunk.symbols,
                resolved_variables: variables,
                resolved_properties: properties
            };
            totalEntries++;
        }
    }

    fs.writeFileSync(registryPath, JSON.stringify(registry, null, 2));
    console.log(`[+] Updated Logic Registry at ${registryPath} with ${totalEntries} bootstrap entries.`);
    console.log(`[+] Total registry size: ${Object.keys(registry).length} entries.`);
}

updateRegistryFromBootstrap().catch(console.error);
