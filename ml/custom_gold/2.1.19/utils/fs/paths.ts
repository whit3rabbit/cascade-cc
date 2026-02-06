import { existsSync, statSync } from "node:fs";
import { join, resolve, dirname, parse } from "node:path";
import { homedir } from "node:os";

/**
 * Checks if a file exists.
 * @param filePath - Path to check.
 * @returns True if exists.
 */
export function fileExists(filePath: string): boolean {
    try {
        return existsSync(filePath);
    } catch {
        return false;
    }
}

/**
 * Gets the current user's home directory.
 * @returns Home directory path.
 */
export function getHomeDir(): string {
    return homedir();
}

/**
 * Joins paths to create a file path.
 * Alias for path.join.
 * @param paths - Paths to join.
 * @returns Joined path.
 */
export function joinPaths(...paths: string[]): string {
    return join(...paths);
}

/**
 * Backward compatible alias for joinPaths.
 * @deprecated Use joinPaths instead.
 */
export const getRelativeFilePath = joinPaths;

import { EnvService } from "../../services/config/EnvService.js";

/**
 * Reads an environment variable.
 * @param key - Environment variable key.
 * @returns Value if found, otherwise undefined.
 */
export function readEnvVar(key: string): string | undefined {
    return EnvService.get(key);
}

/**
 * Resolves a path that might start with ~ to the home directory.
 * @param filePath - Path to resolve.
 * @returns Resolved absolute path.
 */
export function resolvePathFromHome(filePath: string): string {
    if (filePath.startsWith("~")) {
        return join(getHomeDir(), filePath.slice(1));
    }
    return resolve(filePath);
}

/**
 * Checks if a path contains glob characters.
 * @param p - Path to check.
 * @returns True if path has glob markers (*, ?, [, ]).
 */
export function hasGlob(p: string): boolean {
    return p.includes("*") || p.includes("?") || p.includes("[") || p.includes("]");
}

/**
 * Finds the nearest git root by walking up the directory tree.
 * @param startPath - Path to start searching from.
 * @returns Path to the git root directory, or null if not found.
 */
export function findGitRoot(startPath: string): string | null {
    let currentPath = resolve(startPath);
    const root = parse(currentPath).root;

    while (true) {
        const gitPath = join(currentPath, ".git");
        try {
            const stats = statSync(gitPath);
            if (stats.isDirectory() || stats.isFile()) {
                // Determine if it's a file (worktree) or directory (repo)
                // Both are considered root indicators in the original code.
                return currentPath;
            }
        } catch {
            // Ignore if .git doesn't exist or isn't accessible
        }

        if (currentPath === root || currentPath === dirname(currentPath)) {
            break;
        }
        currentPath = dirname(currentPath);
    }

    return null;
}

/**
 * Gets the current project root.
 * Tries to find a git root starting from process.cwd().
 * Defaults to process.cwd() if no git root is found.
 * @returns Project root path.
 */
export function getProjectRoot(): string {
    const cwd = process.cwd();
    const gitRoot = findGitRoot(cwd);
    return gitRoot || cwd;
}
