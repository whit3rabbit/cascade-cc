import fs from "fs";
import path from "path";
import os from "os";
import { marked } from "marked";
import { getProjectRoot } from "../persistence/persistenceUtils.js";
import { getSettings, getGlobalSettings } from "../settings/settingsManager.js";
import ignore from "ignore";

// Logic from chunk_387 (LI5, gY2, pKA, JJ0, etc)

export interface InstructionFile {
    path: string;
    type: "Project" | "Local" | "User" | "Managed";
    content: string;
    globs?: string[];
    parent?: string;
}

const MAX_FILE_SIZE = 40000;
const MAX_DEPTH = 5;

// Parse frontmatter (LI5)
function parseInstructionFile(content: string): { content: string, globs?: string[] } {
    return { content };
}

export function loadInstructionFile(filePath: string, type: "Project" | "Local" | "User" | "Managed"): InstructionFile | null {
    try {
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
        const rawContent = fs.readFileSync(filePath, "utf-8");
        const { content, globs } = parseInstructionFile(rawContent);

        return {
            path: filePath,
            type,
            content,
            globs
        };
    } catch (err: any) {
        if (err.code === 'EACCES') {
            // log permission error
        }
        return null;
    }
}

// Recursive loader (bk)
export function loadInstructionsRecursively(
    filePath: string,
    type: "Project" | "Local" | "User" | "Managed",
    visited: Set<string>,
    approved: boolean,
    depth: number = 0,
    parent?: string
): InstructionFile[] {
    if (visited.has(filePath) || depth >= MAX_DEPTH) return [];

    // Resolving symlinks logic from eX
    let targetPath = filePath;
    try {
        targetPath = fs.realpathSync(filePath);
    } catch { /* ignore */ }

    visited.add(filePath);
    if (filePath !== targetPath) visited.add(targetPath);

    const instruction = loadInstructionFile(filePath, type);
    if (!instruction || !instruction.content.trim()) return [];

    if (parent) instruction.parent = parent;

    const results = [instruction];

    // Find nested includes (OI5)
    // Pass targetPath's directory as baseDir
    const includedPaths = extractIncludes(instruction.content, path.dirname(targetPath));

    for (const includePath of includedPaths) {
        const nested = loadInstructionsRecursively(includePath, type, visited, approved, depth + 1, filePath);
        results.push(...nested);
    }

    return results;
}

function extractIncludes(markdown: string, baseDir: string): string[] {
    const included = new Set<string>();
    const tokens = marked.lexer(markdown);

    const traverse = (tokens: any[]) => {
        for (const token of tokens) {
            if (token.type === 'text') {
                const text = token.text || "";
                // Regex from OI5: /(?:^|\s)@((?:[^\s\\]|\\ )+)/g
                const regex = /(?:^|\s)@((?:[^\s\\]|\\ )+)/g;
                let match;
                while ((match = regex.exec(text)) !== null) {
                    let ref = match[1];
                    if (!ref) continue;
                    ref = ref.replace(/\\ /g, " ");
                    // Path validation logic from OI5
                    if (ref.startsWith("./") || ref.startsWith("~/") || (ref.startsWith("/") && ref !== "/") || ref.match(/^[a-zA-Z0-9._-]/)) {
                        const resolved = path.resolve(baseDir, ref.startsWith("~") ? ref.replace("~", os.homedir()) : ref);
                        included.add(resolved);
                    }
                }
            }
            if (token.tokens) traverse(token.tokens);
            if (token.items) traverse(token.items); // Lists
        }
    };

    traverse(tokens);
    return [...included];
}

// Rules loader (pKA)
export function loadRules(
    rulesDir: string,
    type: "Project" | "Local" | "User" | "Managed",
    visited: Set<string>
): InstructionFile[] {
    if (visited.has(rulesDir)) return [];

    try {
        if (!fs.existsSync(rulesDir) || !fs.statSync(rulesDir).isDirectory()) return [];
        visited.add(rulesDir);

        const files = fs.readdirSync(rulesDir, { withFileTypes: true });
        const results: InstructionFile[] = [];

        for (const file of files) {
            const fullPath = path.join(rulesDir, file.name);
            if (file.isDirectory()) {
                results.push(...loadRules(fullPath, type, visited));
            } else if (file.isFile() && file.name.endsWith(".md")) {
                const instructions = loadInstructionsRecursively(fullPath, type, visited, true);
                results.push(...instructions);
            }
        }
        return results;
    } catch (err) {
        return [];
    }
}

export function getAllInstructions(): string {
    const cwd = process.cwd();
    const instructions: InstructionFile[] = [];
    const visited = new Set<string>();

    // 1. Managed
    // instructions.push(...)

    // 2. User Settings
    // ...

    // 3. Project Settings
    const projectRoot = getProjectRoot(cwd);
    if (getSettings().projectSettings) {
        const claudeMd = path.join(projectRoot, "CLAUDE.md");
        instructions.push(...loadInstructionsRecursively(claudeMd, "Project", visited, false));

        const dotClaudeMd = path.join(projectRoot, ".claude", "CLAUDE.md");
        instructions.push(...loadInstructionsRecursively(dotClaudeMd, "Project", visited, false));

        const rulesDir = path.join(projectRoot, ".claude", "rules");
        instructions.push(...loadRules(rulesDir, "Project", visited));
    }

    // Format output (JJ0)
    if (instructions.length === 0) return "";

    const intro = "Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.";

    const formatted = instructions.map(instr => {
        let label = "";
        if (instr.type === "Project") label = " (project instructions, checked into the codebase)";
        else if (instr.type === "Local") label = " (user's private project instructions, not checked in)";
        else label = " (user's private global instructions for all projects)";

        return `Contents of ${instr.path}${label}:\n\n${instr.content}`;
    }).join("\n\n");

    return `${intro}\n\n${formatted}`;
}
