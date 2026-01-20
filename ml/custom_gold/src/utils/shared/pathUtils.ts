
// Logic from chunk_70.ts (Shell & Path Utilities)

import { homedir } from "node:os";
import { join, resolve, normalize, isAbsolute, dirname } from "node:path";

// --- Custom Error Classes ---

export class AbortError extends Error {
    constructor(message = "The operation was aborted.") {
        super(message);
        this.name = "AbortError";
    }
}

export class ShellError extends Error {
    constructor(public stdout: string, public stderr: string, public code: number, public interrupted: boolean) {
        super("Shell command failed");
        this.name = "ShellError";
    }
}

export class ConfigParseError extends Error {
    constructor(message: string, public filePath: string, public defaultConfig?: any) {
        super(message);
        this.name = "ConfigParseError";
    }
}

// --- Path Utilities ---

/**
 * Resolves paths with support for home directory (~) and Windows normalization.
 */
export function resolvePath(path: string, baseDir: string = process.cwd()): string {
    const trimmed = path.trim();
    if (!trimmed) return normalize(baseDir);
    if (trimmed === "~") return homedir();
    if (trimmed.startsWith("~/")) return join(homedir(), trimmed.slice(2));

    let target = trimmed;
    // Handle Windows /c/ style paths if needed
    if (process.platform === "win32" && target.match(/^\/[a-z]\//i)) {
        target = target[1].toUpperCase() + ":" + target.slice(2);
    }

    if (isAbsolute(target)) return normalize(target);
    return resolve(baseDir, target);
}

export function isPathTraversal(path: string): boolean {
    return /(?:^|[\\/])\.\.(?:[\\/]|$)/.test(path);
}

export function normalizeToPosix(path: string): string {
    return normalize(path).replace(/\\/g, "/");
}

// --- Shell Utilities ---

/**
 * Escapes arguments for shell execution.
 */
export function quoteShellArgs(args: string[]): string {
    return args.map(arg => {
        if (/^[a-zA-Z0-9/_.-]+$/.test(arg)) return arg;
        return `'${arg.replace(/'/g, "'\\''")}'`;
    }).join(" ");
}
export function getProjectRoot() { return process.cwd(); }

export function getConfigDir(): string {
    return process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude");
}

