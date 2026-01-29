/**
 * File: src/utils/fs/pathSanitizer.ts
 * Role: Utilities for sanitizing and validating file paths across platforms.
 */

import { randomBytes } from "node:crypto";

/**
 * Removes surrounding quotes from a string.
 */
export function removeQuotes(str: string): string {
    if (!str) return str;
    const trimmed = str.trim();
    if (
        (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
        (trimmed.startsWith("'") && trimmed.endsWith("'"))
    ) {
        return trimmed.slice(1, -1);
    }
    return trimmed;
}

/**
 * Handles backslashes and escaped characters in a string for non-Windows platforms.
 */
export function handleBackslashes(inputString: string): string {
    if (process.platform === "win32") {
        return inputString;
    }
    const doubleBackslashPlaceholder = `__DOUBLE_BACKSLASH_${randomBytes(8).toString("hex")}__`;
    return inputString
        .replace(/\\\\/g, doubleBackslashPlaceholder)
        .replace(/\\(.)/g, "$1")
        .replace(new RegExp(doubleBackslashPlaceholder, "g"), "\\");
}

/**
 * Checks if a string is a valid file path (basic validation).
 */
export function isValidFilePath(filePath: string): boolean {
    const trimmedPath = removeQuotes(filePath.trim());
    const processedPath = handleBackslashes(trimmedPath);
    // Basic sanity check for characters allowed in common file systems
    return /^[a-zA-Z0-9._\\/\s-]+$/.test(processedPath);
}

/**
 * Sanitizes a file path, ensuring it's valid and safe for use.
 */
export function sanitizeFilePath(filePath: string): string | null {
    const trimmedPath = removeQuotes(filePath.trim());
    const processedPath = handleBackslashes(trimmedPath);
    if (isValidFilePath(processedPath)) {
        return processedPath;
    }
    return null;
}
