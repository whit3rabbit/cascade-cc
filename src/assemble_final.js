const fs = require('fs');
const path = require('path');
const semver = require('semver');

function getLatestVersion(outputRoot) {
    if (!fs.existsSync(outputRoot)) return null;
    const versions = fs.readdirSync(outputRoot).filter(f => {
        const fullPath = path.join(outputRoot, f);
        return fs.statSync(fullPath).isDirectory() && semver.valid(f);
    });

    if (versions.length === 0) return null;

    // Sort versions using semver to get the actual latest
    return versions.sort(semver.rcompare)[0];
}

function getDeobfuscatedChunkPath(chunksDir, chunkMeta) {
    if (!chunkMeta) return null;

    const originalFile = path.basename(chunkMeta.file);
    const chunkBase = path.basename(originalFile, '.js');

    let logicalName = "";
    if (chunkMeta.suggestedFilename) {
        logicalName = chunkMeta.suggestedFilename;
    } else if (chunkMeta.kb_info && chunkMeta.kb_info.suggested_path) {
        logicalName = path.basename(chunkMeta.kb_info.suggested_path.replace(/`/g, ''), '.ts').replace('.js', '');
    }

    // Sanitize logicalName to match rename_chunks.js
    logicalName = logicalName.replace(/[\/\\?%*:|"<>]/g, '_');

    const finalName = logicalName ? `${chunkBase}_${logicalName}.js` : originalFile;
    const fullPath = path.join(chunksDir, finalName);

    if (fs.existsSync(fullPath)) return fullPath;

    // Fallback: Check if the original chunk name exists
    const fallbackPath = path.join(chunksDir, originalFile);
    if (fs.existsSync(fallbackPath)) return fallbackPath;

    return null;
}

async function assemble(version) {
    const outputRoot = './cascade_graph_analysis';

    if (!version) {
        version = getLatestVersion(outputRoot);
        if (!version) {
            console.error(`[!] No versions found in ${outputRoot}`);
            return;
        }
        console.log(`[*] No version specified. Defaulting to latest: ${version}`);
    }

    const versionPath = path.join(outputRoot, version);
    const metadataPath = path.join(versionPath, 'metadata', 'graph_map.json');
    const chunksDir = path.join(versionPath, 'deobfuscated_chunks');
    const finalDir = path.join(versionPath, 'assemble');

    if (!fs.existsSync(metadataPath)) {
        console.error(`[!] Metadata not found: ${metadataPath}`);
        return;
    }

    const graphData = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    const chunks = Array.isArray(graphData) ? graphData : (graphData.chunks || []);

    console.log(`[*] Total chunks loaded from metadata: ${chunks.length}`);

    if (!chunks || chunks.length === 0) {
        console.error(`[!] No chunks found in metadata: ${metadataPath}`);
        return;
    }

    // 1. Identify "Original Modules" and group chunks accordingly
    // An "Original Module" is a sequence of chunks that starts with an import chunk
    // and continues until the next import chunk or the end.
    const modules = [];
    let currentModule = null;

    for (const chunk of chunks) {
        if (chunk.startsWithImport) {
            currentModule = {
                chunks: [chunk],
                bestPath: null
            };
            modules.push(currentModule);
        } else if (currentModule) {
            currentModule.chunks.push(chunk);
        }
    }

    // Identify the best path for each module and mark chunks for inclusion
    const finalChunksToAssemble = new Set();

    modules.forEach(mod => {
        // Find the most descriptive path within this module
        let bestPath = null;
        for (const chunk of mod.chunks) {
            if (chunk.suggestedPath) {
                bestPath = chunk.suggestedPath.replace('.ts', '.js');
            } else if (chunk.kb_info && chunk.kb_info.suggested_path) {
                bestPath = chunk.kb_info.suggested_path.replace(/`/g, '').replace('.ts', '.js');
            }
            if (bestPath) break;
        }

        // Fallback path if no descriptive path found
        if (!bestPath && mod.chunks.length > 0) {
            const firstChunk = mod.chunks[0];
            const folder = firstChunk.role.toLowerCase().replace(/_/g, '-');
            const fileName = firstChunk.suggestedFilename || firstChunk.name;
            bestPath = `src/services/${folder}/${fileName}.js`;
        }

        mod.bestPath = bestPath;

        // Mark all chunks in this "Original Module" for inclusion
        mod.chunks.forEach(c => {
            c.finalPath = bestPath;
            finalChunksToAssemble.add(c);
        });
    });

    // Also include other "Core" chunks that might not be in an original module (rare but possible)
    chunks.forEach(c => {
        if (!finalChunksToAssemble.has(c) && (
            c.category === 'family' ||
            c.category === 'priority' ||
            c.category === 'founder' ||
            (c.category === 'vendor' && c.suggestedFilename)
        )) {
            finalChunksToAssemble.add(c);
        }
    });

    const coreChunks = Array.from(finalChunksToAssemble);

    console.log(`[*] Assembling ${coreChunks.length} chunks into a structured codebase...`);

    // 2. Map chunks to their final file paths
    const fileMap = new Map(); // path -> Array<chunkMetadata>

    for (const chunk of coreChunks) {
        let finalPath = chunk.finalPath;

        if (!finalPath) {
            // 1. Check for Neural/Golden suggested path
            if (chunk.suggestedPath) {
                finalPath = chunk.suggestedPath.replace('.ts', '.js');
            }
            // 2. Check for Knowledge Base path
            else if (chunk.kb_info && chunk.kb_info.suggested_path) {
                finalPath = chunk.kb_info.suggested_path.replace(/`/g, '').replace('.ts', '.js');
            }
            // 3. Fallback to Role-based grouping
            else {
                const folder = chunk.role.toLowerCase().replace(/_/g, '-');
                const fileName = chunk.suggestedFilename || chunk.name;
                finalPath = `src/services/${folder}/${fileName}.js`;
            }
        }

        if (!fileMap.has(finalPath)) fileMap.set(finalPath, []);
        fileMap.get(finalPath).push(chunk);
    }

    // 3. Write Files
    // NOTE: Avoid deleting the entire finalDir to preserve manual changes if any
    if (!fs.existsSync(finalDir)) fs.mkdirSync(finalDir, { recursive: true });

    for (const [filePath, chunkList] of fileMap) {
        const fullOutputPath = path.join(finalDir, filePath);
        const dir = path.dirname(fullOutputPath);

        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        // Sort chunks using a Directed Acyclic Graph (DAG) topological sort with SCC for cycles
        const { Graph } = require('graphology');
        const { stronglyConnectedComponents } = require('graphology-library/components');
        const { topologicalSort } = require('graphology-dag');

        const graph = new Graph({ directed: true });
        chunkList.forEach(c => graph.addNode(c.name));

        chunkList.forEach(chunk => {
            if (chunk.outbound) {
                chunk.outbound.forEach(targetId => {
                    if (graph.hasNode(targetId)) {
                        try {
                            if (!graph.hasDirectedEdge(chunk.name, targetId)) {
                                graph.addDirectedEdge(chunk.name, targetId);
                            }
                        } catch (e) { }
                    }
                });
            }
        });

        // Use SCC to handle cycles
        console.log(`[*] Detecting Strongly Connected Components for ${chunkList.length} chunks...`);
        const sccs = stronglyConnectedComponents(graph);

        // Map node to its SCC index
        const nodeToScc = new Map();
        sccs.forEach((nodes, idx) => {
            nodes.forEach(node => nodeToScc.set(node, idx));
        });

        // Create condensation graph (DAG of SCCs)
        const condensation = new Graph({ directed: true });
        sccs.forEach((_, idx) => condensation.addNode(idx));

        graph.forEachDirectedEdge((source, target) => {
            const sccSource = nodeToScc.get(source);
            const sccTarget = nodeToScc.get(target);
            if (sccSource !== sccTarget && !condensation.hasDirectedEdge(sccTarget, sccSource)) {
                // Dependency: source -> target means target should come AFTER source in assembly
                // BUT wait, topoSort usually gives order where source comes first.
                // In assembly, we want dependencies to be defined before they are used.
                // So if source -> target, target is the dependency. target should come first?
                // Actually, in JS bundles, if module A depends on B, B is often defined after A if it's a lazy load, 
                // but usually dependencies are at the top.
                // The existing logic used .reverse() on topologicalSort.
                if (!condensation.hasDirectedEdge(sccSource, sccTarget)) {
                    condensation.addDirectedEdge(sccSource, sccTarget);
                }
            }
        });

        let sortedNames = [];
        try {
            const sortedSccIndices = topologicalSort(condensation).reverse();
            sortedSccIndices.forEach(idx => {
                const nodesInScc = sccs[idx];
                // Heuristic sort within the SCC
                const sortedNodesInScc = nodesInScc.sort((aName, bName) => {
                    const a = chunkList.find(c => c.name === aName);
                    const b = chunkList.find(c => c.name === bName);
                    if (a.startsWithImport && !b.startsWithImport) return -1;
                    if (!a.startsWithImport && b.startsWithImport) return 1;
                    if ((b.score || 0) !== (a.score || 0)) return (b.score || 0) - (a.score || 0);
                    return (a.startLine || 0) - (b.startLine || 0);
                });
                sortedNames.push(...sortedNodesInScc);
            });
            console.log(`    [*] SCC-based sort successful.`);
        } catch (err) {
            console.warn(`    [!] SCC Topo-sort failed: ${err.message}. Falling back to pure heuristic.`);
            sortedNames = chunkList.slice().sort((a, b) => {
                if (a.startsWithImport && !b.startsWithImport) return -1;
                if (!a.startsWithImport && b.startsWithImport) return 1;
                if (b.outbound && b.outbound.includes(a.name)) return -1;
                if (a.outbound && a.outbound.includes(b.name)) return 1;
                if ((b.score || 0) !== (a.score || 0)) return (b.score || 0) - (a.score || 0);
                return (a.startLine || 0) - (b.startLine || 0);
            }).map(c => c.name);
        }

        const sortedChunks = sortedNames.map(name => chunkList.find(c => c.name === name)).filter(Boolean);

        // FORCE IMPORT SORT: Move chunks with imports to the very top
        // This overrides topological sort because ES import statements must be at the top level
        // and often our graph analysis might infer a dependency direction that places them later.
        const importChunks = [];
        const otherChunks = [];
        sortedChunks.forEach(c => {
            if (c.startsWithImport) {
                importChunks.push(c);
            } else {
                otherChunks.push(c);
            }
        });

        // Re-assemble with imports first - essentially "hoisting" the module header
        const finalSortedChunks = [...importChunks, ...otherChunks];

        const headers = new Set();
        const finalChunks = [];

        for (const chunkMeta of finalSortedChunks) {
            const chunkFilePath = getDeobfuscatedChunkPath(chunksDir, chunkMeta);
            if (chunkFilePath) {
                let code = fs.readFileSync(chunkFilePath, 'utf8');

                // Header Cleaning: Detect and extract common helpers
                const helperRegex = /var (?:__defProp|__export|__toESM|__commonJS|__copyProps|__getProtoOf|__hasOwnProp|__markAsModule|__name|__require|__wrapCommonJS) = [\s\S]*?;(?=\n|$)/g;
                const matches = code.match(helperRegex);
                if (matches) {
                    matches.forEach(m => headers.add(m));
                    code = code.replace(helperRegex, '').trim();
                }

                finalChunks.push({
                    meta: chunkMeta,
                    code
                });
            } else {
                console.warn(`    [!] Missing deobfuscated chunk: ${chunkMeta.name} (Expected at ${chunksDir})`);
            }
        }

        let mergedCode = `/**\n * File: ${filePath}\n * Role: ${chunkList[0].role}\n * Aggregated from ${chunkList.length} chunks\n */\n\n`;

        if (headers.size > 0) {
            mergedCode += `if (!globalThis.__CASCADE_HELPERS_LOADED) {\n`;
            mergedCode += Array.from(headers).join('\n') + '\n';
            mergedCode += `  globalThis.__CASCADE_HELPERS_LOADED = true;\n}\n\n`;
        }

        for (const { meta, code } of finalChunks) {
            mergedCode += `// --- Chunk: ${meta.name} (Original lines: ${meta.startLine}-${meta.endLine}) ---\n`;
            mergedCode += (code || '// (Empty or missing code)') + "\n\n";
        }

        fs.writeFileSync(fullOutputPath, mergedCode);
        console.log(`    [+] Generated: ${filePath}`);
    }


    console.log(`\n[OK] Assembly complete. Codebase located in: ${finalDir}`);
}

const version = process.argv[2];
assemble(version);

