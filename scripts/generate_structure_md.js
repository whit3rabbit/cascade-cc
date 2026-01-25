const fs = require('fs');
const path = require('path');
const { loadKnowledgeBase } = require('../src/knowledge_base');

const OUTPUT_PATH = path.resolve('structrecc.md');

function writeLines(lines, node, depth, dirName) {
    const indent = '  '.repeat(depth);
    if (dirName) {
        lines.push(`${indent}${dirName}/`);
    }

    const files = Array.isArray(node.files) ? node.files.slice() : [];
    files.sort().forEach(file => {
        lines.push(`${indent}  ${file}`);
    });

    Object.entries(node).forEach(([key, value]) => {
        if (key === 'files' || key === 'description') return;
        if (value && typeof value === 'object') {
            writeLines(lines, value, dirName ? depth + 1 : depth, key);
        }
    });
}

function buildStructureMd(projectStructure) {
    const lines = [];
    lines.push('# structrecc');
    lines.push('');
    lines.push('Purpose: Proposed filesystem map for LLM-guided deobfuscation/refinement.');
    lines.push(`Generated: ${new Date().toISOString()}`);
    lines.push('');

    Object.entries(projectStructure || {}).forEach(([rootKey, value]) => {
        if (!value || typeof value !== 'object') return;
        writeLines(lines, value, 0, rootKey);
        lines.push('');
    });

    return lines.join('\n').trimEnd() + '\n';
}

const { kb } = loadKnowledgeBase();
if (!kb || !kb.project_structure) {
    console.error('[!] No project_structure found in knowledge base.');
    process.exit(1);
}

const content = buildStructureMd(kb.project_structure);
fs.writeFileSync(OUTPUT_PATH, content, 'utf8');
console.log(`[*] Wrote ${OUTPUT_PATH}`);
