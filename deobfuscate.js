require('dotenv').config();
const fs = require('fs');
const path = require('path');
const axios = require('axios');

const OUTPUT_ROOT = './cascade_graph_analysis';

// --- KNOWLEDGE BASE ---
const KB_PATH = './knowledge_base.json';
let KB = null;
if (fs.existsSync(KB_PATH)) {
    KB = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
    console.log(`[*] Loaded Knowledge Base with ${KB.name_hints?.length || 0} name hints.`);
}

// --- LLM CONFIG ---
const PROVIDER = process.env.LLM_PROVIDER || 'gemini'; // 'gemini' or 'openrouter'
const MODEL = process.env.LLM_MODEL || (PROVIDER === 'gemini' ? 'gemini-2.0-flash' : 'google/gemini-2.0-flash-exp:free');
const API_KEY = PROVIDER === 'gemini' ? process.env.GEMINI_API_KEY : process.env.OPENROUTER_API_KEY;

if (!API_KEY) {
    console.error(`[!] Error: No API key found for provider ${PROVIDER}.`);
    console.error(`    Please set ${PROVIDER === 'gemini' ? 'GEMINI_API_KEY' : 'OPENROUTER_API_KEY'} in .env`);
    process.exit(1);
}

// --- UTILS ---
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function callLLM(prompt, retryCount = 0) {
    try {
        let response;
        if (PROVIDER === 'gemini') {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${API_KEY}`;
            response = await axios.post(url, {
                contents: [{ parts: [{ text: prompt }] }]
            });
            return response.data.candidates[0].content.parts[0].text;
        } else {
            // OpenRouter
            response = await axios.post('https://openrouter.ai/api/v1/chat/completions', {
                model: MODEL,
                messages: [{ role: 'user', content: prompt }]
            }, {
                headers: {
                    'Authorization': `Bearer ${API_KEY}`,
                    'HTTP-Referer': 'https://github.com/whit3rabbit/cascade-like',
                    'X-Title': 'Cascade-Like Deobfuscator'
                }
            });
            return response.data.choices[0].message.content;
        }
    } catch (err) {
        if (err.response && err.response.status === 429 && retryCount < 5) {
            const delay = Math.pow(2, retryCount) * 2000 + Math.random() * 1000;
            console.warn(`[!] Rate limited (429). Retrying in ${Math.round(delay / 1000)}s...`);
            await sleep(delay);
            return callLLM(prompt, retryCount + 1);
        }
        throw err;
    }
}

function getLatestVersion(baseDir) {
    if (!fs.existsSync(baseDir)) return null;
    const dirs = fs.readdirSync(baseDir).filter(f => fs.statSync(path.join(baseDir, f)).isDirectory());

    // Filter folders that look like semver (e.g., 2.1.3)
    const semverDirs = dirs.filter(d => /^\d+\.\d+\.\d+/.test(d));
    if (semverDirs.length === 0) return null;

    // Sort semver-like strings correctly
    semverDirs.sort((a, b) => {
        const partsA = a.split('.').map(Number);
        const partsB = b.split('.').map(Number);
        for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
            const valA = partsA[i] || 0;
            const valB = partsB[i] || 0;
            if (valA !== valB) return valB - valA; // Descending
        }
        return 0;
    });

    return semverDirs[0];
}

// --- MAIN STAGES ---
async function run() {
    const version = process.argv[2] || getLatestVersion(OUTPUT_ROOT);
    if (!version) {
        console.error(`[!] Error: No version found to deobfuscate.`);
        process.exit(1);
    }

    const versionPath = path.join(OUTPUT_ROOT, version);
    const chunksDir = path.join(versionPath, 'chunks');
    const mappingPath = path.join(versionPath, 'mapping.json');
    const deobfuscatedDir = path.join(versionPath, 'deobfuscated_chunks');

    if (!fs.existsSync(chunksDir)) {
        console.error(`[!] Error: Chunks directory not found at ${chunksDir}`);
        process.exit(1);
    }

    if (!fs.existsSync(deobfuscatedDir)) fs.mkdirSync(deobfuscatedDir, { recursive: true });

    let mapping = {};
    if (fs.existsSync(mappingPath)) {
        mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
        console.log(`[*] Resuming with existing mapping (${Object.keys(mapping).length} entries)`);
    }

    const graphMapPath = path.join(versionPath, 'metadata', 'graph_map.json');
    let graphData = [];
    if (fs.existsSync(graphMapPath)) {
        graphData = JSON.parse(fs.readFileSync(graphMapPath, 'utf8'));
    }

    const chunkFiles = fs.readdirSync(chunksDir).filter(f => f.endsWith('.js')).sort();

    console.log(`[*] Starting Deobfuscation for version: ${version}`);
    console.log(`[*] Found ${chunkFiles.length} chunks.`);

    // Stage 1: Incremental Mapping
    for (let i = 0; i < chunkFiles.length; i++) {
        const file = chunkFiles[i];
        const code = fs.readFileSync(path.join(chunksDir, file), 'utf8');

        // Find metadata for this chunk
        const chunkMeta = graphData.find(m => m.file.endsWith(file));
        const contextInfo = chunkMeta ? `
CHUNK CONTEXT:
- Role: ${chunkMeta.role}
- Label: ${chunkMeta.label}
- Category: ${chunkMeta.category}
- State Touchpoints: ${chunkMeta.state_touchpoints.join(', ')}
- Neighbors (Calls/Is Called By): ${chunkMeta.outbound.join(', ')}
` : '';

        console.log(`[*] Stage 1 [${i + 1}/${chunkFiles.length}]: Analyzing ${file}...`);

        const prompt = `
You are an expert JavaScript deobfuscator. Your task is to identify obfuscated variable names, function names, and property names in the provided code chunk and create a mapping to more meaningful names.

${contextInfo}

${KB && KB.name_hints ? `
REFERENCE HINTS FROM PREVIOUS VERSIONS:
${KB.name_hints.map(h => `- If the code performs this logic: "${h.logic_anchor}", use the name: "${h.suggested_name.replace(/`/g, '')}"`).join('\n')}
` : ''}

CURRENT MAPPING (JSON):
${JSON.stringify(mapping, null, 2)}

CODE CHUNK:
\`\`\`javascript
${code}
\`\`\`

INSTRUCTIONS:
1. Examine the code chunk for obfuscated names (e.g., 'a', 'b_1', 'q_0', etc.).
2. Since the role is '${chunkMeta.role}', prioritize names related to that:
   - If 'STREAM_ORCHESTRATOR', names like 'stream', 'event', 'payload' are likely.
   - If 'SHELL_EXECUTOR', names like 'command', 'args', 'stdout' are likely.
   - If 'API_CLIENT', names like 'endpoint', 'headers', 'request' are likely.
3. Use the following State DNA touchpoints to identify state-related variables:
   ${chunkMeta.state_touchpoints.map(tp => `- Access to '${tp}' suggests logic managing ${tp.replace(/([A-Z])/g, ' $1')}`).join('\n')}
4. If a name is already in the CURRENT MAPPING, use it consistently.
5. If the logic matches a REFERENCE HINT above, you MUST use that suggested name.
6. Return ONLY a JSON object containing the NEW mappings you found in this chunk. Do not include the old mappings unless you are updating them with better context.

Format your response as a single JSON object:
{
  "obfuscatedName": "descriptiveName",
  ...
}
`;

        try {
            const llmResponse = await callLLM(prompt);
            // Extract JSON from response (sometimes models wrap in markdown)
            const jsonMatch = llmResponse.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const newMappings = JSON.parse(jsonMatch[0]);
                Object.assign(mapping, newMappings);
                fs.writeFileSync(mappingPath, JSON.stringify(mapping, null, 2));
                console.log(`    - Added ${Object.keys(newMappings).length} new mappings.`);
            } else {
                console.warn(`    [!] Could not parse JSON from LLM response for ${file}`);
            }
        } catch (err) {
            console.error(`    [!] LLM Call failed for ${file}: ${err.message}`);
        }
    }

    console.log(`[*] Stage 1 Complete. Final mapping preserved at ${mappingPath}`);

    // Stage 2: Final Rewrite
    console.log(`[*] Starting Stage 2: Rewriting chunks with consistent names...`);
    for (let i = 0; i < chunkFiles.length; i++) {
        const file = chunkFiles[i];
        const code = fs.readFileSync(path.join(chunksDir, file), 'utf8');

        console.log(`[*] Stage 2 [${i + 1}/${chunkFiles.length}]: Rewriting ${file}...`);

        const prompt = `
You are an expert JavaScript deobfuscator. Your task is to rewrite the provided code chunk using the provided name mapping.

FINAL MAPPING (JSON):
${JSON.stringify(mapping, null, 2)}

CODE CHUNK:
\`\`\`javascript
${code}
\`\`\`

INSTRUCTIONS:
1. Replace all occurrences of the obfuscated names in the FINAL MAPPING with their corresponding descriptive names.
2. Ensure the resulting code is valid JavaScript.
3. Keep the original structure and logic of the code intact.
4. Return ONLY the rewritten JavaScript code. No markdown, no comments.

Rewritten Code:
`;

        try {
            const rewrittenCode = await callLLM(prompt);
            // Clean up backticks if any
            const cleanedCode = rewrittenCode.replace(/^```javascript\n/, '').replace(/```$/, '').trim();
            fs.writeFileSync(path.join(deobfuscatedDir, file), cleanedCode);
            console.log(`    - Saved rewritten chunk to ${path.join('deobfuscated_chunks', file)}`);
        } catch (err) {
            console.error(`    [!] LLM Call failed for ${file}: ${err.message}`);
        }
    }

    console.log(`\n[COMPLETE] Deobfuscation finished for ${version}`);
}

run().catch(console.error);
