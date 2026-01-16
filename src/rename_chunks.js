const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

const RESERVED_PROPERTIES = new Set([
    'then', 'catch', 'finally', 'length', 'map', 'forEach', 'filter', 'reduce',
    'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'join', 'split',
    'includes', 'indexOf', 'lastIndexOf', 'hasOwnProperty', 'toString',
    'valueOf', 'prototype', 'constructor', 'apply', 'call', 'bind',
    'message', 'stack', 'name', 'code', 'status', 'headers', 'body'
]);

/**
 * Safely renames identifiers in a piece of code using Babel's scope-aware renaming.
 */
function renameIdentifiers(code, mapping, sourceInfo = {}) {
    const { sourceFile = null, neighbors = [], displayName = null, suggestedPath = null } = sourceInfo;
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx']
        });

        const renamedBindings = new Set();
        const chunkBase = sourceFile ? path.basename(String(sourceFile), '.js') : null;

        traverse(ast, {
            Identifier(p) {
                const oldName = p.node.name;

                // 1. Variable Renaming (Scope-aware)
                if (mapping.variables && Object.prototype.hasOwnProperty.call(mapping.variables, oldName) && !renamedBindings.has(oldName)) {
                    const binding = p.scope.getBinding(oldName);
                    if (binding) {
                        const isTopLevel = binding.scope.block.type === 'Program';
                        const isGlobalConstant = binding.scope.depth <= 2 && binding.constant;

                        if (isTopLevel || isGlobalConstant) {
                            const entry = mapping.variables[oldName];
                            if (!entry) return;

                            let newName = null;
                            if (Array.isArray(entry)) {
                                // Prioritize exact chunk match
                                const match = entry.find(e => e && (e.source === chunkBase || e.source === sourceFile)) ||
                                    entry.find(e => e && (neighbors.includes(e.source) || neighbors.includes(path.basename(String(e.source), '.js')))) ||
                                    (displayName && entry.find(e => e && e.source && e.source.includes(displayName))) ||
                                    entry[0];
                                newName = match ? match.name : null;
                            } else {
                                newName = typeof entry === 'string' ? entry : (entry ? entry.name : null);
                            }

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
                    let confidence = 0.5;

                    if (Array.isArray(entry)) {
                        const chunkMatch = entry.find(e => e && (e.source === chunkBase || e.source === sourceFile));
                        if (chunkMatch) {
                            newName = chunkMatch.name;
                            confidence = chunkMatch.confidence || 0.8;
                        } else {
                            const neighborMatch = entry.find(e => e && (neighbors.includes(e.source) || neighbors.includes(path.basename(String(e.source), '.js'))));
                            if (neighborMatch) {
                                newName = neighborMatch.name;
                                confidence = neighborMatch.confidence || 0.8;
                            } else if (displayName || suggestedPath) {
                                const contextMatch = entry.find(e =>
                                    (displayName && e.source && e.source.includes(displayName)) ||
                                    (suggestedPath && e.source && suggestedPath.includes(e.source))
                                );
                                if (contextMatch) {
                                    newName = contextMatch.name;
                                    confidence = contextMatch.confidence || 0.7;
                                }
                            }

                            // Fallback to absolute highest confidence only if prop name is descriptive
                            if (!newName) {
                                const highConf = entry.find(e => e && e.confidence >= 0.98);
                                if (highConf && propName.length > 4) {
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
                        // SAFETY CHECKS:
                        const isObjectKnown = (p.node.object.type === 'Identifier' && mapping.variables[p.node.object.name]) ||
                            p.node.object.type === 'ThisExpression' ||
                            p.node.object.type === 'MemberExpression'; // Allow nested properties if they look logical
                        const isHighConfidence = confidence >= 0.95;
                        const isDescriptive = propName.length > 2;

                        if (RESERVED_PROPERTIES.has(propName)) {
                            // SKIP: Never rename reserved properties
                        } else if (isObjectKnown || (isHighConfidence && isDescriptive)) {
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
                    let confidence = 0.5;

                    if (Array.isArray(entry)) {
                        const chunkMatch = entry.find(e => e && (e.source === chunkBase || e.source === sourceFile));
                        if (chunkMatch) {
                            newName = chunkMatch.name;
                            confidence = chunkMatch.confidence || 0.8;
                        } else {
                            const neighborMatch = entry.find(e => e && (neighbors.includes(e.source) || neighbors.includes(path.basename(String(e.source), '.js'))));
                            if (neighborMatch) {
                                newName = neighborMatch.name;
                                confidence = neighborMatch.confidence || 0.8;
                            } else if (displayName || suggestedPath) {
                                const contextMatch = entry.find(e =>
                                    (displayName && e.source && e.source.includes(displayName)) ||
                                    (suggestedPath && e.source && suggestedPath.includes(e.source))
                                );
                                if (contextMatch) {
                                    newName = contextMatch.name;
                                    confidence = contextMatch.confidence || 0.7;
                                }
                            }
                        }
                    } else {
                        newName = typeof entry === 'string' ? entry : (entry ? entry.name : null);
                        confidence = (entry && typeof entry === 'object') ? (entry.confidence || 0.8) : 0.8;
                    }

                    if (newName) {
                        const isHighConfidence = confidence >= 0.95;
                        const isDescriptive = propName.length > 3;

                        if (RESERVED_PROPERTIES.has(propName)) {
                            // SKIP
                        } else if (isHighConfidence || isDescriptive) {
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
            const renamedCode = renameIdentifiers(code, mapping, {
                sourceFile: chunkBase,
                neighbors,
                displayName: chunkMeta?.displayName,
                suggestedPath: chunkMeta?.kb_info?.suggested_path
            });

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
