/**
 * File: src/utils/shared/formatUri.ts
 * Role: Formats URIs (usually file://) from LSP servers or other tools for terminal display.
 */

import { relative } from "node:path";
import { terminalLog } from "./runtime.js";

/**
 * Formats a URI into a cleaner, human-readable path.
 * 
 * @param uri - The URI to format (e.g., 'file:///path/to/file').
 * @param workspaceRoot - Optional root directory to make the path relative to.
 * @returns A formatted path string.
 */
export function formatUri(uri: string | undefined, workspaceRoot?: string): string {
    if (!uri) {
        terminalLog("Warning: formatUri called with undefined URI", "warn");
        return "<unknown location>";
    }

    // Remove file:// prefix
    let path = uri.replace(/^file:\/\//, "");

    // Path cleanup for different OS (especially Windows vs Unix)
    if (process.platform === 'win32') {
        // Basic Windows file path cleanup if needed
        path = path.replace(/^\/([a-z]):/i, "$1:");
    }

    try {
        path = decodeURIComponent(path);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        terminalLog(`Warning: Failed to decode LSP URI '${uri}': ${message}. Using un-decoded path.`, "warn");
    }

    if (workspaceRoot) {
        try {
            const relativePath = relative(workspaceRoot, path);
            // Use relative path only if it's practically helpful
            if (relativePath.length < path.length && !relativePath.startsWith("../../")) {
                return relativePath;
            }
        } catch {
            // If relative conversion fails, just return absolute path
        }
    }

    return path;
}

/**
 * Formats an LSP location (URI + Range) into a 'file:line:char' string.
 */
export function formatLocation(uri: string, line: number, character: number, workspaceRoot?: string): string {
    const formattedUri = formatUri(uri, workspaceRoot);
    return `${formattedUri}:${line + 1}:${character + 1}`;
}

/**
 * Groups a list of items by their URI.
 */
export function groupResultsByUri<T extends { uri: string }>(items: T[], workspaceRoot?: string): Map<string, T[]> {
    const grouped = new Map<string, T[]>();
    for (const item of items) {
        const formatted = formatUri(item.uri, workspaceRoot);
        const list = grouped.get(formatted) || [];
        list.push(item);
        grouped.set(formatted, list);
    }
    return grouped;
}

// Aliases for deobfuscated code compatibility
export const cgA = formatUri;
export const Cs7 = groupResultsByUri;
export const Xj1 = (location: { uri: string, range: { start: { line: number, character: number } } }, root?: string) =>
    formatLocation(location.uri, location.range.start.line, location.range.start.character, root);
