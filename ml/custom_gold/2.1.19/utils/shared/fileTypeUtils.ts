/**
 * File: src/utils/shared/fileTypeUtils.ts
 * Role: Logic for identifying specific Claude session file types.
 */

import { homedir } from 'node:os';

/**
 * Determines the file type ("session_memory" or "session_transcript") from a path.
 */
export function determineFileType(filePath: string): string | null {
    const normalizedPath = filePath.replace(/\\/g, '/');
    const home = homedir().replace(/\\/g, '/');

    if (!normalizedPath.startsWith(home)) {
        return null;
    }

    if (normalizedPath.includes("/session-memory/") && normalizedPath.endsWith(".md")) {
        return "session_memory";
    }

    if (normalizedPath.includes("/projects/") && (normalizedPath.endsWith(".jsonl") || normalizedPath.endsWith(".json"))) {
        return "session_transcript";
    }

    return null;
}

/**
 * Determines the file type from a glob pattern.
 */
export function determineFileTypeFromGlob(pattern: string): string | null {
    const normalizedPattern = pattern.replace(/\\/g, '/');

    if (normalizedPattern.includes("session-memory") && (normalizedPattern.includes(".md") || normalizedPattern.endsWith("*"))) {
        return "session_memory";
    }

    if (normalizedPattern.includes(".jsonl") || (normalizedPattern.includes("projects") && normalizedPattern.includes("*.jsonl"))) {
        return "session_transcript";
    }

    return null;
}
