
import { execShellCommand } from "./shellUtils.js";

/**
 * Checks if the Git working directory is clean.
 * Deobfuscated from T0A in chunk_227.ts.
 */
export async function isGitClean(): Promise<boolean> {
    const { stdout } = await execShellCommand("git", ["status", "--porcelain"]);
    return stdout.trim().length === 0;
}

/**
 * Returns the current Git branch name.
 * Deobfuscated from CJ1 in chunk_443.ts / dg in chunk_227.ts.
 */
export async function gitGetCurrentBranch(): Promise<string> {
    const { stdout } = await execShellCommand("git", ["branch", "--show-current"]);
    const branch = stdout.trim();
    if (branch) return branch;

    // Fallback if --show-current is not available (older git)
    const { stdout: headInfo } = await execShellCommand("git", ["rev-parse", "--abbrev-ref", "HEAD"]);
    return headInfo.trim();
}

/**
 * Parses a Git remote URL into "owner/repo" format.
 * Deobfuscated logic from M63 in chunk_227.ts / bBA in chunk_333.ts.
 */
export function parseGitRepoName(url: string): string | null {
    const trimmed = url.trim();
    if (!trimmed) return null;

    // SSH format: git@github.com:owner/repo.git
    const sshMatch = trimmed.match(/^git@[^:]+:(.+?)(?:\.git)?$/);
    if (sshMatch && sshMatch[1]) {
        return sshMatch[1].toLowerCase();
    }

    // HTTPS format: https://github.com/owner/repo.git
    const httpsMatch = trimmed.match(/^(?:https?|ssh):\/\/(?:[^@]+@)?[^/]+\/(.+?)(?:\.git)?$/);
    if (httpsMatch && httpsMatch[1]) {
        return httpsMatch[1].toLowerCase();
    }

    return null;
}

/**
 * Returns the repository name (owner/repo) from origin remote.
 * Deobfuscated from WP in chunk_334.ts / H11 in chunk_227.ts.
 */
export async function getGitRepoName(): Promise<string | null> {
    const { stdout, code } = await execShellCommand("git", ["remote", "get-url", "origin"]);
    if (code !== 0) return null;
    return parseGitRepoName(stdout.trim());
}
