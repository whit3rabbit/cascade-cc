import * as path from "node:path";
import { runRipgrep } from "../../utils/shared/ripgrep.js";

/**
 * Files and directories to always hide/protect.
 * Deobfuscated from qLA, X93 in chunk_220.ts.
 */

export const PROTECTED_FILES = [
    ".gitconfig",
    ".gitmodules",
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".ripgreprc",
    ".mcp.json"
];

export const PROTECTED_DIRS = [
    ".git",
    ".vscode",
    ".idea",
    ".claude/commands",
    ".claude/agents"
];

/**
 * Scans for sensitive files in the project to hide from the sandbox.
 * Deobfuscated from W93 in chunk_220.ts.
 */
export async function scanSensitiveFiles(
    projectRoot: string = process.cwd(),
    maxDepth: number = 3,
    allowGitConfig: boolean = false,
    abortSignal?: AbortSignal
): Promise<string[]> {
    const hiddenFiles = new Set<string>();

    // Add static protected paths
    for (const f of PROTECTED_FILES) hiddenFiles.add(path.resolve(projectRoot, f));
    for (const d of PROTECTED_DIRS) hiddenFiles.add(path.resolve(projectRoot, d));
    hiddenFiles.add(path.resolve(projectRoot, ".git/hooks"));
    if (!allowGitConfig) hiddenFiles.add(path.resolve(projectRoot, ".git/config"));

    // Build ripgrep glob arguments to find matching files
    const args: string[] = ["--files", "--hidden", "--max-depth", String(maxDepth)];
    for (const f of PROTECTED_FILES) {
        args.push("--iglob", f);
    }
    for (const d of PROTECTED_DIRS) {
        args.push("--iglob", `**/${d}/**`);
    }
    args.push("--iglob", "**/.git/hooks/**");
    if (!allowGitConfig) args.push("--iglob", "**/.git/config");
    args.push("-g", "!**/node_modules/**");

    try {
        const matches = await runRipgrep([projectRoot], "", abortSignal, { args });
        for (const match of matches) {
            const fullPath = path.resolve(projectRoot, match);

            // Heuristic: if it's inside a protected dir like .git, hide the whole dir
            let foundDir = false;
            for (const dirName of [...PROTECTED_DIRS, ".git"]) {
                const segments = fullPath.split(path.sep);
                const idx = segments.findIndex(s => s.toLowerCase() === dirName.toLowerCase());
                if (idx !== -1) {
                    const basePath = segments.slice(0, idx + 1).join(path.sep);
                    if (dirName === ".git") {
                        if (fullPath.includes(".git/hooks")) hiddenFiles.add(path.join(basePath, "hooks"));
                        else if (fullPath.includes(".git/config")) hiddenFiles.add(path.join(basePath, "config"));
                    } else {
                        hiddenFiles.add(basePath);
                    }
                    foundDir = true;
                    break;
                }
            }

            if (!foundDir) hiddenFiles.add(fullPath);
        }
    } catch (err) {
        // Log error if debug enabled
    }

    return Array.from(hiddenFiles);
}
