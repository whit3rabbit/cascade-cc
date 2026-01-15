const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

/**
 * Safely renames identifiers in a piece of code using Babel's scope-aware renaming.
 */
function renameIdentifiers(code, mapping, sourceFile = null, neighbors = []) {
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx']
        });

        const renamedBindings = new Set();

        traverse(ast, {
            Identifier(p) {
                const oldName = p.node.name;

                // 1. Variable Renaming (Scope-aware)
                if (mapping.variables && Object.prototype.hasOwnProperty.call(mapping.variables, oldName) && !renamedBindings.has(oldName)) {
                    const binding = p.scope.getBinding(oldName);
                    if (binding) {
                        // Refined scope check: Renaming if Program level OR in a shallow wrapper (common in esbuild)
                        // This allows renaming module-level globals that are wrapped in __commonJS routines.
                        const isTopLevel = binding.scope.block.type === 'Program';
                        const isShallow = binding.scope.depth <= 2;

                        if (isTopLevel || isShallow) {
                            const entry = mapping.variables[oldName];
                            if (!entry) return;

                            // If multiple sources exist, prioritize the one matching current file or its neighbors
                            const newName = Array.isArray(entry)
                                ? (
                                    entry.find(e => e && e.source === sourceFile)?.name ||
                                    entry.find(e => e && neighbors.includes(e.source))?.name ||
                                    (entry[0] ? entry[0].name : null)
                                )
                                : (typeof entry === 'string' ? entry : (entry ? entry.name : null));

                            if (newName) {
                                p.scope.rename(oldName, newName);
                                renamedBindings.add(oldName);
                            }
                        }
                    }
                }
            },
            MemberExpression(p) {
                // 2. Property Renaming (Direct Member Access: obj.prop)
                if (p.node.computed) return;

                const propName = p.node.property.name;
                if (mapping.properties && Object.prototype.hasOwnProperty.call(mapping.properties, propName)) {
                    const entry = mapping.properties[propName];
                    if (!entry) return;
                    let newName = null;
                    let confidence = 0.8;

                    if (Array.isArray(entry)) {
                        const chunkMatch = entry.find(e => e && e.source === path.basename(String(sourceFile)));
                        if (chunkMatch) {
                            newName = chunkMatch.name;
                            confidence = chunkMatch.confidence || 0.8;
                        } else {
                            const neighborMatch = entry.find(e => e && neighbors.includes(e.source));
                            if (neighborMatch) {
                                newName = neighborMatch.name;
                                confidence = neighborMatch.confidence || 0.8;
                            } else {
                                const highConf = entry.find(e => e && e.confidence >= 0.9);
                                if (highConf) {
                                    newName = highConf.name;
                                    confidence = highConf.confidence;
                                }
                            }
                        }
                    } else {
                        newName = typeof entry === 'string' ? entry : (entry ? entry.name : null);
                        confidence = (entry && typeof entry === 'object') ? (entry.confidence || 0.8) : 0.8;
                    }

                    if (newName) {
                        // SAFETY CHECKS: Allow renaming if object is known OR it's a 'this' expression
                        const isObjectKnown = (p.node.object.type === 'Identifier' && mapping.variables[p.node.object.name]) ||
                            p.node.object.type === 'ThisExpression';
                        const isHighConfidence = confidence >= 0.98;
                        const isUnique = propName.length > 2;

                        if (isObjectKnown || isHighConfidence || isUnique) {
                            p.node.property.name = newName;
                        }
                    }
                }
            },
            ObjectProperty(p) {
                // 3. Object Property Renaming ({ prop: value })
                if (p.node.computed) return;

                const propName = p.node.key.name;
                if (mapping.properties && Object.prototype.hasOwnProperty.call(mapping.properties, propName)) {
                    const entry = mapping.properties[propName];
                    if (!entry) return;
                    let newName = null;
                    let confidence = 0.8;

                    if (Array.isArray(entry)) {
                        const chunkMatch = entry.find(e => e && e.source === path.basename(String(sourceFile)));
                        if (chunkMatch) {
                            newName = chunkMatch.name;
                            confidence = chunkMatch.confidence || 0.8;
                        } else {
                            const neighborMatch = entry.find(e => e && neighbors.includes(e.source));
                            if (neighborMatch) {
                                newName = neighborMatch.name;
                                confidence = neighborMatch.confidence || 0.8;
                            } else {
                                const highConf = entry.find(e => e && e.confidence >= 0.9);
                                if (highConf) {
                                    newName = highConf.name;
                                    confidence = highConf.confidence;
                                }
                            }
                        }
                    } else {
                        newName = typeof entry === 'string' ? entry : (entry ? entry.name : null);
                        confidence = (entry && typeof entry === 'object') ? (entry.confidence || 0.8) : 0.8;
                    }

                    if (newName) {
                        // SAFETY CHECKS: For object properties, we are slightly more lenient if it's high confidence or unique
                        const isHighConfidence = confidence >= 0.98;
                        const isUnique = propName.length > 2;

                        if (isHighConfidence || isUnique) {
                            p.node.key.name = newName;
                        }
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
            const outputPath = path.join(deobfuscatedDir, finalName);

            const neighbors = chunkMeta ? [...chunkMeta.neighbors || [], ...chunkMeta.outbound || []] : [];
            const renamedCode = renameIdentifiers(code, mapping, chunkBase, neighbors);

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
                const fallbackPath = path.join(deobfuscatedDir, file);
                fs.writeFileSync(fallbackPath, fs.readFileSync(path.join(chunksDir, file), 'utf8'));
            } catch (e) { }
            failCount++;
        }
    }

    console.log(`[OK] Stage 2 Complete: ${successCount} succeeded, ${failCount} failed.`);
}

main().catch(console.error);
