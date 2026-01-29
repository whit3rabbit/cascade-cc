/**
 * File: src/utils/shared/displayUtils.ts
 * Role: General-purpose utilities for formatting time, text, and other display values.
 */

/**
 * Formats a duration from a start timestamp into a human-readable string (e.g., "1h 20m 30s").
 * 
 * @param startTime - The starting timestamp in milliseconds.
 * @returns {string} The formatted duration string.
 */
export function formatDuration(startTime: number): string {
    const totalSeconds = Math.floor((Date.now() - startTime) / 1000);
    if (totalSeconds < 0) return "0s";

    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    const parts: string[] = [];
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0 || hours > 0) parts.push(`${minutes}m`);
    parts.push(`${seconds}s`);

    return parts.join(" ");
}

/**
 * Simplistic truncation helper.
 * @param text - The text to truncate.
 * @param maxLength - Maximum allowed length.
 * @returns {string} Truncated string with ellipsis.
 */
export function truncate(text: string, maxLength: number): string {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + "...";
}

/**
 * Increments a numeric value.
 */
export function increment(value: number): number {
    return value + 1;
}
