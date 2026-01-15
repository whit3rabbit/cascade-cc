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
    const chunks = Array.isArray(graphData) ? graphData : graphData.chunks;

    if (!chunks) {
        console.error(`[!] No chunks found in metadata: ${metadataPath}`);
        return;
    }

    // 1. Filter for "Core" chunks (Skip vendor)
    const coreChunks = chunks.filter(c => c.category === 'family' || c.category === 'priority' || c.category === 'founder');

    console.log(`[*] Assembling ${coreChunks.length} core chunks into a structured codebase...`);

    // 2. Map chunks to their final file paths
    const fileMap = new Map(); // path -> Array<chunkMetadata>

    for (const chunk of coreChunks) {
        let finalPath = '';

        if (chunk.kb_info && chunk.kb_info.suggested_path) {
            // Priority 1: Knowledge Base Path
            finalPath = chunk.kb_info.suggested_path.replace(/`/g, '').replace('.ts', '.js');
        } else if (chunk.suggestedFilename) {
            // Priority 2: LLM Suggested Name
            const folder = chunk.role.toLowerCase().replace('_', '-');
            finalPath = `src/services/${folder}/${chunk.suggestedFilename}.js`;
        } else {
            // Priority 3: Fallback based on role
            const folder = chunk.role.toLowerCase().replace('_', '-');
            finalPath = `src/undetermined/${folder}/${chunk.name}.js`;
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

        // Sort chunks using a Directed Acyclic Graph (DAG) topological sort
        const { Graph } = require('graphology');
        const { topologicalSort } = require('graphology-dag');

        const graph = new Graph({ directed: true });
        chunkList.forEach(c => graph.addNode(c.name));

        chunkList.forEach(chunk => {
            if (chunk.outbound) {
                chunk.outbound.forEach(targetId => {
                    if (graph.hasNode(targetId)) {
                        try {
                            graph.addDirectedEdge(chunk.name, targetId);
                        } catch (e) {
                            // Circular dependency or existing edge, skip
                        }
                    }
                });
            }
        });

        let sortedNames;
        try {
            // graphology's topologicalSort returns nodes in order such that for every edge u -> v, u comes before v.
            // In our case, if 'a' depends on 'b' (a -> b), we might want 'b' first for imports,
            // but for logical flow, we usually want 'a' before 'b' if 'a' is the caller.
            // The original logic was: `if (b.outbound && b.outbound.includes(a.name)) return -1;`
            // This means if 'b' points to 'a', 'a' comes first.
            // So we want the REVERSE of a standard topological sort if we consider outbound as "calls".
            // Actually, for imports/dependencies, if B is a neighbor of A (A -> B), B should likely come after A OR before A depending on context.
            // Standard topological sort: for edge u -> v, u comes before v.
            // If A -> B (A depends on B), then A comes before B.
            // This is usually what we want for logical flow (High level -> Low level).
            // Reverse standard topological sort: for edge u -> v (u depends on v), v should come before u.
            // Actually, in our graph, edges are Chunk -> Target (Dependent -> Dependency).
            // topologicalSort(graph) returns [Dependent, Dependency].
            // We want [Dependency, Dependent].
            sortedNames = topologicalSort(graph).reverse();
        } catch (e) {
            // If there's a cycle, fall back to the original heuristic
            console.warn(`    [!] Cycle detected in ${filePath}, falling back to heuristic sort.`);
            sortedNames = chunkList.sort((a, b) => {
                if (b.outbound && b.outbound.includes(a.name)) return -1;
                if (a.outbound && a.outbound.includes(b.name)) return 1;
                if (a.startsWithImport && !b.startsWithImport) return -1;
                if (!a.startsWithImport && b.startsWithImport) return 1;
                return a.startLine - b.startLine;
            }).map(c => c.name);
        }

        const sortedChunks = sortedNames.map(name => chunkList.find(c => c.name === name)).filter(Boolean);

        // Secondary stable sort for non-dependent chunks (e.g. imports first)
        // Note: sortedChunks already respects DAG order. We only want to bubble up imports if it doesn't break DAG.
        // For simplicity, we'll stick to the DAG sort primarily.

        let mergedCode = `/**\n * File: ${filePath}\n * Role: ${chunkList[0].role}\n * Aggregated from ${chunkList.length} chunks\n */\n\n`;

        for (const chunkMeta of sortedChunks) {
            const chunkFilePath = getDeobfuscatedChunkPath(chunksDir, chunkMeta);

            if (chunkFilePath) {
                const code = fs.readFileSync(chunkFilePath, 'utf8');
                mergedCode += `// --- Chunk: ${chunkMeta.name} (Original lines: ${chunkMeta.startLine}-${chunkMeta.endLine}) ---\n`;
                mergedCode += code + "\n\n";
            } else {
                console.warn(`    [!] Missing deobfuscated chunk: ${chunkMeta.name} (Expected at ${chunksDir})`);
            }
        }

        fs.writeFileSync(fullOutputPath, mergedCode);
        console.log(`    [+] Generated: ${filePath}`);
    }


    console.log(`\n[OK] Assembly complete. Codebase located in: ${finalDir}`);
}

const version = process.argv[2];
assemble(version);

