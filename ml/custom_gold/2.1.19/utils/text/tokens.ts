/**
 * File: src/utils/text/tokens.ts
 * Role: Utilities for string truncation and tokenization.
 */

import { stringWidth } from './width.js';

export interface TruncateOptions {
    position?: 'start' | 'middle' | 'end';
    truncationCharacter?: string;
}

/**
 * Truncates a string to the given visual length.
 * 
 * @param {string} text - The string to truncate.
 * @param {number} length - The maximum allowed visual length.
 * @param {TruncateOptions} options - Truncation configuration.
 * @returns {string} The truncated string.
 */
export function truncateString(text: string, length: number, options: TruncateOptions = {}): string {
    if (!text || text.length === 0) return "";
    const { position = 'end', truncationCharacter = '…' } = options;

    const width = stringWidth(text);
    if (width <= length) return text;

    const charWidth = stringWidth(truncationCharacter);
    const limit = Math.max(0, length - charWidth);

    if (position === 'start') {
        // Basic implementation: take the last 'limit' chars
        return truncationCharacter + text.slice(-limit);
    } else if (position === 'middle') {
        const half = Math.floor(limit / 2);
        return text.slice(0, half) + truncationCharacter + text.slice(-half);
    }

    // end (default)
    return text.slice(0, limit) + truncationCharacter;
}
