/**
 * File: src/utils/fs/paths.ts
 * Role: Common file system path utilities.
 */

import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
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
 * Gets the current project root.
 * Defaults to process.cwd() if not otherwise specified.
 * @returns Project root path.
 */
export function getProjectRoot(): string {
    return process.cwd();
}
