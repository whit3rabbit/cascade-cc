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
    if (chunkMeta.proposedPath) {
        logicalName = path.basename(chunkMeta.proposedPath, '.ts');
    } else if (chunkMeta.suggestedFilename) {
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

    // 1. Group chunks by their INTENDED logical file path
    // Previous logic relied on 'startsWithImport', but now we trust the LLM/Classifier's 'suggestedPath'
    const modulesMap = new Map(); // path -> Array<chunk>

    const getChunkRole = chunk => (typeof chunk.role === 'string' ? chunk.role : '');

    chunks.forEach(chunk => {
        let logicalPath = null;

        // Priority 1: Proven Vendor
        if (chunk.category === 'vendor') {
            const role = getChunkRole(chunk) || 'vendor_misc';
            const folder = role.replace(/[:\s]/g, '_').replace(/VENDOR_/g, '').toLowerCase();
            const fileName = chunk.suggestedFilename || chunk.name;
            logicalPath = `src/vendor/${folder}/${fileName}.js`;
        }
        // Priority 2: Explicit Proposal
        else if (chunk.proposedPath) {
            logicalPath = chunk.proposedPath;
        }
        // Priority 3: LLM Suggested Path
        else if (chunk.suggestedPath) {
            logicalPath = chunk.suggestedPath;
        }
        // Priority 4: KB Legacy
        else if (chunk.kb_info && chunk.kb_info.suggested_path) {
            logicalPath = chunk.kb_info.suggested_path.replace(/`/g, '');
        }

        // Clean up path
        if (logicalPath) {
            logicalPath = logicalPath.replace('.ts', '.js');
            if (!logicalPath.startsWith('src/') && !logicalPath.startsWith('test/')) {
                logicalPath = path.join('src', logicalPath);
            }
        } else {
            // Fallback: Group by Role/Folder
            const folder = (getChunkRole(chunk) || 'misc').toLowerCase().replace(/[^a-z0-9]/g, '-');
            const fileName = chunk.suggestedFilename || chunk.name;
            logicalPath = `src/services/${folder}/${fileName}.js`;
        }

        // Sanitization
        logicalPath = logicalPath.replace(/\.\./g, '__').replace(/^\//, '');

        if (!modulesMap.has(logicalPath)) {
            modulesMap.set(logicalPath, []);
        }
        modulesMap.get(logicalPath).push(chunk);
    });

    const finalChunksToAssemble = new Set();
    const modules = [];

    for (const [bestPath, chunkList] of modulesMap) {
        modules.push({
            bestPath,
            chunks: chunkList
        });
        chunkList.forEach(c => {
            c.finalPath = bestPath;
            finalChunksToAssemble.add(c);
        });
    }

    // Also include other "Core" chunks that might not be in an original module (rare but possible)
    chunks.forEach(c => {
        if (!finalChunksToAssemble.has(c) && (
            c.category === 'family' ||
            c.category === 'priority' ||
            c.category === 'founder'
            // Exclude vendor chunks as per user request
            // || (c.category === 'vendor' && c.suggestedFilename)
        )) {
            finalChunksToAssemble.add(c);
        }
    });

    const coreChunks = Array.from(finalChunksToAssemble).filter(c =>
        c.category !== 'vendor' &&
        !getChunkRole(c).startsWith('LIB:') &&
        !getChunkRole(c).toLowerCase().startsWith('lib:') &&
        !c.matchIsLibrary &&
        !(c.finalPath && (c.finalPath.includes('/vendor/') || c.finalPath.includes('/third_party/'))) &&
        !(c.proposedPath && (c.proposedPath.includes('/vendor/') || c.proposedPath.includes('/third_party/')))
    );

    console.log(`[*] Assembling ${coreChunks.length} chunks into a structured codebase...`);

    // 2. Map chunks to their final file paths
    const fileMap = new Map(); // path -> Array<chunkMetadata>

    for (const chunk of coreChunks) {
        // Since we already calculated finalPath in step 1, we just need to verify and add
        let finalPath = chunk.finalPath;

        // DOUBLE CHECK: Ensure no path ever points to vendor if it's not a vendor chunk (extra safety)
        if (chunk.category !== 'vendor' && finalPath.includes('/vendor/')) {
            finalPath = finalPath.replace('/vendor/', '/third_party/');
        }

        if (!finalPath) {
            console.warn(`[WARN] Chunk ${chunk.name} has no final path? defaulting.`);
            finalPath = `src/services/misc/${chunk.name}.js`;
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

                // Module Wrapper Stripping (Hard Signal)
                // If this chunk starts a module wrapper (search for __commonJS( or __lazyInit(), strip the envelope
                const wrapperStartRegex = /var\s+[\w$]+\s*=\s*(?:__commonJS|__lazyInit)\s*\(((?:exports)?\s*=>\s*\{|\s*function\s*\(\w*\)\s*\{)/;
                if (wrapperStartRegex.test(code)) {
                    // console.log(`    [i] Stripping module wrapper from ${chunkMeta.name}`);
                    code = code.replace(wrapperStartRegex, '');
                }

                // Strip wrapper end
                const wrapperEndRegex = /\}\);\s*$/;
                if (wrapperEndRegex.test(code)) {
                    code = code.replace(wrapperEndRegex, '');
                }

                finalChunks.push({
                    meta: chunkMeta,
                    code
                });
            } else {
                console.warn(`    [!] Missing deobfuscated chunk: ${chunkMeta.name} (Expected at ${chunksDir})`);
            }
        }

        let mergedCode = `/**\n * File: ${filePath}\n * Role: ${getChunkRole(chunkList[0]) || 'unknown'}\n * Aggregated from ${chunkList.length} chunks\n */\n\n`;

        if (headers.size > 0) {
            mergedCode += `/* CASCADE HELPERS (De-duplicated) */\n`;
            mergedCode += `if (!globalThis.__CASCADE_HELPERS_LOADED) {\n`;
            mergedCode += Array.from(headers).join('\n') + '\n';
            mergedCode += `  globalThis.__CASCADE_HELPERS_LOADED = true;\n}\n\n`;
        }

        for (const { meta, code } of finalChunks) {
            mergedCode += `// --- Chunk: ${meta.name} (Original lines: ${meta.startLine}-${meta.endLine}) ---\n`;
            if (code && code.trim()) {
                mergedCode += code + "\n\n";
            }
        }

        fs.writeFileSync(fullOutputPath, mergedCode);
        console.log(`    [+] Generated: ${filePath}`);
    }

    // 4. Generate Assembly Map
    console.log(`[*] Generating Assembly Map...`);
    let mapContent = `# Assembly Map\n\nGenerated on: ${new Date().toISOString()}\n\n`;

    // Sort files alphabetically for the report
    const sortedFiles = Array.from(fileMap.keys()).sort();

    for (const filePath of sortedFiles) {
        const chunkList = fileMap.get(filePath);
        mapContent += `## ${filePath}\n\n`;

        // Sort chunks by start line for consistent output
        const sortedChunksForMap = chunkList.slice().sort((a, b) => (a.startLine || 0) - (b.startLine || 0));

        for (const chunk of sortedChunksForMap) {
            const originalFileName = path.basename(chunk.file);
            mapContent += `- **${chunk.name}**\n`;
            mapContent += `  - Original File: \`${originalFileName}\`\n`;
            mapContent += `  - Lines: ${chunk.startLine}-${chunk.endLine}\n`;
            mapContent += `  - Role: ${getChunkRole(chunk) || 'unknown'}\n`;
        }
        mapContent += `\n`;
    }

    const mapPath = path.join(finalDir, 'ASSEMBLY_MAP.md');
    fs.writeFileSync(mapPath, mapContent);
    console.log(`    [+] Generated: ASSEMBLY_MAP.md`);


    console.log(`\n[OK] Assembly complete. Codebase located in: ${finalDir}`);
}

const version = process.argv[2];
assemble(version);
