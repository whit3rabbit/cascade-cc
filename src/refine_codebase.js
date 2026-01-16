require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { callLLM, validateKey, PROVIDER, MODEL } = require('./llm_client');
const pLimit = require('p-limit');

const OUTPUT_ROOT = './cascade_graph_analysis';

async function refineFile(filePath, relPath) {
    const code = fs.readFileSync(filePath, 'utf8');

    // Skip very large files or vendor files if needed, but for now let's try all
    if (code.length > 50000) {
        console.warn(`[!] Skipping ${relPath} (too large: ${code.length} chars)`);
        return;
    }

    const prompt = `
Role: Senior Staff Software Engineer / Reverse Engineer
Task: Source Code Reconstruction & Logic Refinement

I have an assembled JavaScript file that was deobfuscated from a minified bundle.
The identifiers names are mostly correct, but the logic structure might still be "minified" (e.g., flattened loops, complex ternary chains, inlined constants, dead code branches).

GOAL: Reconstruct this file into what the ORIGINAL source code likely looked like.

INSTRUCTIONS:
1. Restore clean control flow (use if/else instead of complex ternaries where appropriate).
2. Group related functions/variables logically.
3. Remove any remaining obfuscation artifacts (like proxy functions or unused helper calls).
4. Ensure the exports match a modern ESM/CommonJS structure.
5. Add helpful comments explaining complex logic blocks.
6. FIX any obviously broken logic caused by the assembly process (e.g. out-of-order definitions if detected).

FILE PATH: ${relPath}

CODE:
\`\`\`javascript
${code}
\`\`\`

RESPONSE:
Output ONLY the refined JavaScript code. Do not include markdown blocks or preamble.
`;

    console.log(`[*] Refining ${relPath}...`);
    try {
        const refinedCode = await callLLM(prompt);
        // Remove markdown wrappers if the LLM ignored the instruction
        let cleaned = refinedCode.trim();
        if (cleaned.startsWith('```javascript')) {
            cleaned = cleaned.replace(/^```javascript\n/, '').replace(/\n```$/, '');
        } else if (cleaned.startsWith('```')) {
            cleaned = cleaned.replace(/^```\n/, '').replace(/\n```$/, '');
        }

        const refinedPath = filePath.replace('/assemble/', '/refined_assemble/');
        const refinedDir = path.dirname(refinedPath);
        if (!fs.existsSync(refinedDir)) fs.mkdirSync(refinedDir, { recursive: true });

        fs.writeFileSync(refinedPath, cleaned);
        console.log(`[+] Saved refined version: ${refinedPath}`);
    } catch (err) {
        console.error(`[!] Error refining ${relPath}: ${err.message}`);
    }
}

async function run() {
    let version = process.argv[2];
    if (!version) {
        console.error("Usage: node src/refine_codebase.js <version>");
        process.exit(1);
    }

    const assembleDir = path.join(OUTPUT_ROOT, version, 'assemble');
    if (!fs.existsSync(assembleDir)) {
        console.error(`[!] Assemble directory not found: ${assembleDir}`);
        process.exit(1);
    }

    const isValid = await validateKey();
    if (!isValid) process.exit(1);

    const files = [];
    function walk(dir) {
        fs.readdirSync(dir).forEach(file => {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                walk(fullPath);
            } else if (file.endsWith('.js')) {
                files.push(fullPath);
            }
        });
    }
    walk(assembleDir);

    console.log(`[*] Found ${files.length} files to refine.`);

    const limit = pLimit(PROVIDER === 'gemini' ? 2 : 5);
    const tasks = files.map(file => limit(() => refineFile(file, path.relative(assembleDir, file))));

    await Promise.all(tasks);
    console.log(`[*] Refinement complete.`);
}

run().catch(console.error);
