const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

/**
 * Safely renames identifiers in a piece of code using Babel's scope-aware renaming.
 */
function renameIdentifiers(code, mapping) {
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx'] // Add plugins if needed for other syntaxes
        });

        traverse(ast, {
            Identifier(path) {
                const oldName = path.node.name;
                if (mapping[oldName]) {
                    const newName = mapping[oldName];
                    // Use path.scope.rename to safely rename all occurrences in the scope
                    // This handles variable shadowing and other scope-related complexities
                    path.scope.rename(oldName, newName);
                }
            }
        });

        return generate(ast, {
            retainLines: true,
            compact: false
        }).code;
    } catch (err) {
        console.error(`[!] Babel error: ${err.message}`);
        return null;
    }
}

async function main() {
    const versionPath = process.argv[2];
    if (!versionPath) {
        console.error("Usage: node rename_chunks.js <versionPath>");
        process.exit(1);
    }

    const chunksDir = path.join(versionPath, 'chunks');
    const mappingPath = path.join(versionPath, 'metadata', 'mapping.json');
    const deobfuscatedDir = path.join(versionPath, 'deobfuscated_chunks');

    if (!fs.existsSync(mappingPath)) {
        console.error(`[!] mapping.json not found at ${mappingPath}`);
        process.exit(1);
    }

    const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
    const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js'));

    if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

    console.log(`[*] Renaming ${chunkFiles.length} chunks...`);

    for (const file of chunkFiles) {
        const inputPath = path.join(chunksDir, file);
        const outputPath = path.join(deobfuscatedDir, file);
        const code = fs.readFileSync(inputPath, 'utf8');

        // Note: For large mappings, we might want to optimize this, 
        // but path.scope.rename is already quite efficient.
        const renamedCode = renameIdentifiers(code, mapping);

        if (renamedCode) {
            fs.writeFileSync(outputPath, renamedCode);
            // console.log(`    - Renamed ${file}`);
        } else {
            console.error(`    [!] Failed to rename ${file}, copying original.`);
            fs.writeFileSync(outputPath, code);
        }
    }

    console.log(`[OK] Stage 2: Babel renaming complete.`);
}

main().catch(console.error);
