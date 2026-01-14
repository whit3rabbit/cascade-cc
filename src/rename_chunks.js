const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

/**
 * Safely renames identifiers in a piece of code using Babel's scope-aware renaming.
 */
function renameIdentifiers(code, mapping, sourceFile = null) {
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx']
        });

        const renamedBindings = new Set();

        traverse(ast, {
            Identifier(path) {
                const oldName = path.node.name;

                // 1. Variable Renaming (Scope-aware)
                if (mapping.variables && Object.prototype.hasOwnProperty.call(mapping.variables, oldName) && !renamedBindings.has(oldName)) {
                    const binding = path.scope.getBinding(oldName);
                    if (binding) {
                        // Refined scope check: Renaming if Program level OR in a shallow wrapper (common in esbuild)
                        // This allows renaming module-level globals that are wrapped in __commonJS routines.
                        const isTopLevel = binding.scope.block.type === 'Program';
                        const isShallow = binding.scope.depth <= 2;

                        if (isTopLevel || isShallow) {
                            const entry = mapping.variables[oldName];

                            // If multiple sources exist, prioritize the one matching current file
                            const newName = Array.isArray(entry)
                                ? (entry.find(e => e.source === sourceFile)?.name || entry[0].name)
                                : (typeof entry === 'string' ? entry : (entry ? entry.name : null));

                            if (newName) {
                                path.scope.rename(oldName, newName);
                                renamedBindings.add(oldName);
                            }
                        }
                    }
                }
            },
            MemberExpression(path) {
                // 2. Property Renaming (Direct Member Access: obj.prop)
                if (path.node.computed) return;

                const propName = path.node.property.name;
                if (mapping.properties && Object.prototype.hasOwnProperty.call(mapping.properties, propName)) {
                    const entry = mapping.properties[propName];
                    let newName = null;

                    if (Array.isArray(entry)) {
                        // Priority 1: Current chunk's suggestion
                        const chunkMatch = entry.find(e => e.source === path.basename(sourceFile));
                        if (chunkMatch) {
                            newName = chunkMatch.name;
                        } else {
                            // Priority 2: High confidence common name (if any)
                            // For properties, we are MORE CAUTIOUS. Only rename if it's high confidence.
                            const highConf = entry.find(e => e.confidence >= 0.9);
                            if (highConf) newName = highConf.name;
                        }
                    } else {
                        newName = typeof entry === 'string' ? entry : (entry ? entry.name : null);
                    }

                    if (newName) {
                        path.node.property.name = newName;
                    }
                }
            },
            ObjectProperty(path) {
                // 3. Object Property Renaming ({ prop: value })
                if (path.node.computed) return;

                const propName = path.node.key.name;
                if (mapping.properties && Object.prototype.hasOwnProperty.call(mapping.properties, propName)) {
                    const entry = mapping.properties[propName];
                    let newName = null;

                    if (Array.isArray(entry)) {
                        const chunkMatch = entry.find(e => e.source === path.basename(sourceFile));
                        if (chunkMatch) {
                            newName = chunkMatch.name;
                        } else {
                            const highConf = entry.find(e => e.confidence >= 0.9);
                            if (highConf) newName = highConf.name;
                        }
                    } else {
                        newName = typeof entry === 'string' ? entry : (entry ? entry.name : null);
                    }

                    if (newName) {
                        path.node.key.name = newName;
                    }
                }
            }
        });

        return generate(ast, {
            retainLines: false,
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
            const renamedCode = renameIdentifiers(code, mapping, chunkBase);

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
