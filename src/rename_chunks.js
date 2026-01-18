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
    'message', 'stack', 'name', 'code', 'status', 'headers', 'body',
    'write', 'end', 'on', 'once', 'emit', 'removeListener', 'removeAllListeners',
    'substring', 'substr', 'replace', 'trim', 'toLowerCase', 'toUpperCase', 'charAt',
    'match', 'search', 'slice', 'concat', 'entries', 'keys', 'values', 'from',
    'stdout', 'stderr', 'stdin', 'destroyed', 'preInit'
]);

const DESCRIPTOR_KEYS = new Set([
    'value', 'enumerable', 'configurable', 'writable', 'get', 'set'
]);

const RESERVED_GLOBALS = new Set([
    'Array', 'Object', 'String', 'Number', 'Boolean', 'Function', 'Symbol', 'BigInt',
    'Math', 'JSON', 'Date', 'RegExp', 'Error', 'TypeError', 'Promise', 'Map', 'Set',
    'WeakMap', 'WeakSet', 'Proxy', 'Reflect', 'Intl', 'URL', 'URLSearchParams',
    'TextEncoder', 'TextDecoder', 'Buffer', 'process', 'console', 'global',
    'globalThis', 'window', 'document', 'navigator', 'setTimeout', 'setInterval',
    'clearTimeout', 'clearInterval', 'setImmediate', 'clearImmediate'
]);

const DISALLOWED_VARIABLE_NAMES = new Set([
    'setMinutes'
]);

const isProtectedProperty = (propName) =>
    RESERVED_PROPERTIES.has(propName) || DESCRIPTOR_KEYS.has(propName);

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
                                if (RESERVED_GLOBALS.has(newName) || DISALLOWED_VARIABLE_NAMES.has(newName)) return;
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
                if (!propName) return;
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

                        // SURGICAL RENAME: If we are in surgical mode (which we are by default now for Gold Standard), 
                        // we ONLY apply if it's high confidence.
                        if (isProtectedProperty(propName)) {
                            // SKIP: Never rename reserved properties
                        } else if (isHighConfidence || (isObjectKnown && isDescriptive)) {
                            p.node.property.name = newName;
                        }
                    }
                }
            },
            ObjectProperty(p) {
                // 3. Object Property Renaming ({ prop: value })
                if (p.node.computed) return;

                const propName = p.node.key.name;
                if (!propName) return;
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

                        if (isProtectedProperty(propName)) {
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

const crypto = require('crypto');

async function main() {
    let versionDir = process.argv[2];

    if (!versionDir) {
        // Auto-detect the latest version from cascade_graph_analysis
        const baseDir = './cascade_graph_analysis';
        const dirs = fs.readdirSync(baseDir).filter(d => {
            return fs.statSync(path.join(baseDir, d)).isDirectory() && d !== 'bootstrap';
        }).sort().reverse();

        if (dirs.length > 0) {
            versionDir = path.join(baseDir, dirs[0]);
            console.log(`[*] No version directory specified. Auto-detected latest: ${versionDir}`);
        } else {
            console.log("Usage: node src/rename_chunks.js <version_dir>");
            process.exit(1);
        }
    }

    const chunksDir = path.join(versionDir, 'chunks');
    const mappingPath = path.join(versionDir, 'metadata', 'mapping.json');
    const graphMapPath = path.join(versionDir, 'metadata', 'graph_map.json');
    const deobfuscatedDir = path.join(versionDir, 'deobfuscated_chunks');

    if (!fs.existsSync(mappingPath)) {
        console.error(`[!] mapping.json not found at ${mappingPath}`);
        process.exit(1);
    }

    const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
    const graphDataRaw = fs.existsSync(graphMapPath) ? JSON.parse(fs.readFileSync(graphMapPath, 'utf8')) : [];
    const graphData = Array.isArray(graphDataRaw) ? graphDataRaw : (graphDataRaw.chunks || []);
    const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js'));

    if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

    // Track all files in the deobfuscated directory before processing
    const existingDeobfFiles = new Set(fs.readdirSync(deobfuscatedDir).filter(f => f.endsWith('.js')));
    const generatedFiles = new Set();

    console.log(`[*] Synchronizing ${chunkFiles.length} deobfuscated chunks...`);

    let successCount = 0;
    let failCount = 0;
    let newCount = 0;
    let updatedCount = 0;
    let unchangedCount = 0;

    for (const file of chunkFiles) {
        try {
            const inputPath = path.join(chunksDir, file);
            const code = fs.readFileSync(inputPath, 'utf8');

            // Find metadata for this chunk
            const chunkMeta = graphData.find(m => m.name === path.basename(file, '.js'));

            let logicalName = "";
            if (chunkMeta) {
                if (chunkMeta.proposedPath) {
                    logicalName = path.basename(chunkMeta.proposedPath, '.ts');
                } else if (chunkMeta.suggestedFilename) {
                    logicalName = chunkMeta.suggestedFilename;
                } else if (chunkMeta.kb_info && chunkMeta.kb_info.suggested_path) {
                    logicalName = path.basename(chunkMeta.kb_info.suggested_path.replace(/`/g, ''), '.ts').replace('.js', '');
                }
            }

            logicalName = logicalName.replace(/[\/\\?%*:|"<>]/g, '_');
            const chunkBase = path.basename(file, '.js');
            const finalName = logicalName ? `${chunkBase}_${logicalName}.js` : file;
            const outputPath = path.join(deobfuscatedDir, finalName);
            generatedFiles.add(finalName);

            const neighbors = chunkMeta ? [...(chunkMeta.neighbors || []), ...(chunkMeta.outbound || [])] : [];
            const renamedCode = renameIdentifiers(code, mapping, {
                sourceFile: chunkBase,
                neighbors,
                displayName: chunkMeta?.displayName,
                suggestedPath: chunkMeta?.proposedPath || chunkMeta?.kb_info?.suggested_path
            });

            const finalCode = renamedCode || code;
            const newHash = crypto.createHash('md5').update(finalCode).digest('hex');

            if (fs.existsSync(outputPath)) {
                const oldHash = crypto.createHash('md5').update(fs.readFileSync(outputPath, 'utf8')).digest('hex');
                if (newHash === oldHash) {
                    unchangedCount++;
                } else {
                    fs.writeFileSync(outputPath, finalCode);
                    updatedCount++;
                    console.log(`    [UPDATE] ${finalName}`);
                }
            } else {
                fs.writeFileSync(outputPath, finalCode);
                newCount++;
                console.log(`    [NEW] ${finalName}`);
            }

            if (renamedCode) successCount++;
            else failCount++;

        } catch (err) {
            console.error(`    [!] Error processing ${file}: ${err.message}`);
            failCount++;
        }
    }

    // --- CLEANUP STALE FILES ---
    let deleteCount = 0;
    for (const existingFile of existingDeobfFiles) {
        if (!generatedFiles.has(existingFile)) {
            const filePath = path.join(deobfuscatedDir, existingFile);
            fs.unlinkSync(filePath);
            deleteCount++;
            console.log(`    [DELETE] ${existingFile}`);
        }
    }

    console.log(`\n[COMPLETE] Sync Summary:`);
    console.log(`    - New:       ${newCount}`);
    console.log(`    - Updated:   ${updatedCount}`);
    console.log(`    - Unchanged: ${unchangedCount}`);
    console.log(`    - Deleted:   ${deleteCount}`);
    console.log(`    - Total:     ${generatedFiles.size} chunks in deobfuscated_chunks/`);
    if (failCount > 0) console.warn(`    - Failed to transform ${failCount} chunks (used original code).`);
}

main().catch(console.error);
