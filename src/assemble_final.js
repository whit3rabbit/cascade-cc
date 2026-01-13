const fs = require('fs');
const path = require('path');

async function assemble(version) {
    const outputRoot = './cascade_graph_analysis';
    const versionPath = path.join(outputRoot, version);
    const metadataPath = path.join(versionPath, 'metadata', 'graph_map.json');
    const chunksDir = path.join(versionPath, 'deobfuscated_chunks');
    const finalDir = path.join(versionPath, 'final_codebase');

    if (!fs.existsSync(metadataPath)) {
        console.error(`[!] Metadata not found: ${metadataPath}`);
        return;
    }

    const graphData = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    const chunks = graphData.chunks;

    // 1. Filter for "Core" chunks (Skip vendor)
    const coreChunks = chunks.filter(c => c.category === 'family' || c.category === 'priority');

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

        // Sort chunks by original startLine to preserve logic sequence
        chunkList.sort((a, b) => a.startLine - b.startLine);

        let mergedCode = `/**\n * File: ${filePath}\n * Role: ${chunkList[0].role}\n * Aggregated from ${chunkList.length} chunks\n */\n\n`;

        for (const chunkMeta of chunkList) {
            const chunkFileName = path.basename(chunkMeta.file);
            const chunkFilePath = path.join(chunksDir, chunkFileName);

            if (fs.existsSync(chunkFilePath)) {
                const code = fs.readFileSync(chunkFilePath, 'utf8');
                mergedCode += `// --- Chunk: ${chunkMeta.name} (Original lines: ${chunkMeta.startLine}-${chunkMeta.endLine}) ---\n`;
                mergedCode += code + "\n\n";
            }
        }

        fs.writeFileSync(fullOutputPath, mergedCode);
        console.log(`    [+] Generated: ${filePath}`);
    }

    console.log(`\n[OK] Assembly complete. Codebase located in: ${finalDir}`);
}

const version = process.argv[2];
if (!version) {
    console.log("Usage: node src/assemble_final.js <version>");
} else {
    assemble(version);
}
