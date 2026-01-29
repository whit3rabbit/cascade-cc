/**
 * File: src/utils/fs/fileProcessor.ts
 * Role: File processing utilities for reading and basic parsing.
 */

import { readFileSync } from 'node:fs';

/**
 * Reads the content of a file as a UTF-8 string.
 */
export function readFileContent(filePath: string): string | null {
    try {
        return readFileSync(filePath, 'utf8');
    } catch (error) {
        return null;
    }
}

/**
 * Attempts to parse content as JSON, otherwise returns the original string.
 */
export function parseContent(content: string | null, _safe?: boolean): any {
    if (!content) {
        return null;
    }

    try {
        return JSON.parse(content);
    } catch {
        return content;
    }
}

// --- Aliases for compatibility ---
export {
    readFileContent as yX,
    parseContent as S9
};
