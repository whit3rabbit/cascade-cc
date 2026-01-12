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
                // Use Object.prototype.hasOwnProperty.call to safely check if oldName is a key in mapping
                // This prevents picking up inherited properties like 'toString' or 'hasOwnProperty'
                if (Object.prototype.hasOwnProperty.call(mapping, oldName)) {
                    const newName = mapping[oldName];
                    if (typeof newName === 'string') {
                        path.scope.rename(oldName, newName);
                    }
                }
            }
        });

        return generate(ast, {
            retainLines: true,
            compact: false
        }).code;
    } catch (err) {
        throw new Error(`Babel transform error: ${err.message}`);
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
    const graphMapPath = path.join(versionPath, 'metadata', 'graph_map.json');
    const deobfuscatedDir = path.join(versionPath, 'deobfuscated_chunks');

    if (!fs.existsSync(mappingPath)) {
        console.error(`[!] mapping.json not found at ${mappingPath}`);
        process.exit(1);
    }

    const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
    const graphData = fs.existsSync(graphMapPath) ? JSON.parse(fs.readFileSync(graphMapPath, 'utf8')) : [];
    const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js'));

    if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

    console.log(`[*] Renaming ${chunkFiles.length} chunks...`);

    let successCount = 0;
    let failCount = 0;

    for (const file of chunkFiles) {
        try {
            const inputPath = path.join(chunksDir, file);
            const code = fs.readFileSync(inputPath, 'utf8');

            // Find metadata for this chunk
            const chunkMeta = graphData.find(m => path.basename(m.file) === file);

            let logicalName = "";
            if (chunkMeta) {
                if (chunkMeta.suggestedFilename) {
                    logicalName = chunkMeta.suggestedFilename;
                } else if (chunkMeta.kb_info && chunkMeta.kb_info.suggested_path) {
                    // Extract basename from suggested_path (e.g., `src/utils/foo.ts` -> foo)
                    logicalName = path.basename(chunkMeta.kb_info.suggested_path.replace(/`/g, ''), '.ts').replace('.js', '');
                }
            }

            // Sanitize logicalName: replace slashes and other dangerous characters with underscore
            logicalName = logicalName.replace(/[\/\\?%*:|"<>]/g, '_');

            const chunkBase = path.basename(file, '.js');
            const finalName = logicalName ? `${chunkBase}_${logicalName}.js` : file;
            const outputPath = path.join(deobfuscatedDir, finalName);

            const renamedCode = renameIdentifiers(code, mapping);

            if (renamedCode) {
                fs.writeFileSync(outputPath, renamedCode);
                successCount++;
            } else {
                console.error(`    [!] Failed to transform ${file}, copying original.`);
                fs.writeFileSync(outputPath, code);
                failCount++;
            }
        } catch (err) {
            console.error(`    [!] Error processing ${file}: ${err.message}`);
            // Attempt to write original as fallback
            try {
                const chunkBase = path.basename(file, '.js');
                fs.writeFileSync(path.join(deobfuscatedDir, file), fs.readFileSync(path.join(chunksDir, file), 'utf8'));
            } catch (e) { }
            failCount++;
        }
    }

    console.log(`[OK] Stage 2 Complete: ${successCount} succeeded, ${failCount} failed.`);
}

main().catch(console.error);
