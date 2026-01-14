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
    if (fs.existsSync(finalDir)) fs.rmSync(finalDir, { recursive: true });

    for (const [filePath, chunkList] of fileMap) {
        const fullOutputPath = path.join(finalDir, filePath);
        const dir = path.dirname(fullOutputPath);

        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        // Sort chunks by startsWithImport first, then by original startLine
        chunkList.sort((a, b) => {
            if (a.startsWithImport && !b.startsWithImport) return -1;
            if (!a.startsWithImport && b.startsWithImport) return 1;
            return a.startLine - b.startLine;
        });

        let mergedCode = `/**\n * File: ${filePath}\n * Role: ${chunkList[0].role}\n * Aggregated from ${chunkList.length} chunks\n */\n\n`;

        for (const chunkMeta of chunkList) {
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

