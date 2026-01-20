
// Logic from chunk_70.ts (Windows Bash Detection)

import { execSync } from "node:child_process";
import { join } from "node:path";
import { existsSync } from "node:fs";

/**
 * Finds the path to git-bash on Windows.
 */
export function findGitBashPath(): string | null {
    if (process.platform !== "win32") return null;

    if (process.env.CLAUDE_CODE_GIT_BASH_PATH && existsSync(process.env.CLAUDE_CODE_GIT_BASH_PATH)) {
        return process.env.CLAUDE_CODE_GIT_BASH_PATH;
    }

    try {
        const whereGit = execSync("where.exe git", { stdio: "pipe", encoding: "utf8" }).trim().split("\n")[0];
        if (whereGit) {
            const bashPath = join(whereGit, "..", "..", "bin", "bash.exe");
            if (existsSync(bashPath)) return bashPath;
        }
    } catch {
        // Ignore
    }

    // Common installation paths
    const commonPaths = [
        "C:\\Program Files\\Git\\bin\\bash.exe",
        "C:\\Program Files (x86)\\Git\\bin\\bash.exe"
    ];
    for (const p of commonPaths) {
        if (existsSync(p)) return p;
    }

    return null;
}

/**
 * Converts a Windows path to a Unix/Cygwin path if running in Git Bash.
 */
export function toUnixPath(winPath: string): string {
    if (process.platform !== "win32") return winPath;
    try {
        const bash = findGitBashPath();
        if (!bash) return winPath;
        return execSync(`cygpath -u "${winPath}"`, { shell: bash }).toString().trim();
    } catch {
        return winPath.replace(/\\/g, "/");
    }
}
