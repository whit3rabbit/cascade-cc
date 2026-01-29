/**
 * File: src/utils/text/width.ts
 * Role: Utilities for calculating the visual width of strings, accounting for ANSI codes and CJK characters.
 */

import { stripAnsi } from './ansi.js';

/**
 * Calculates the visual width of a string.
 * Uses Intl.Segmenter if available for accurate grapheme clustering.
 * 
 * @param {string} text - The string to measure.
 * @returns {number} The visual width.
 */
export function stringWidth(text: string): number {
    if (!text || text.length === 0) return 0;

    // First strip ANSI codes
    const plain = stripAnsi(text);

    if (typeof Intl !== 'undefined' && (Intl as any).Segmenter) {
        const segmenter = new (Intl as any).Segmenter('en', { granularity: 'grapheme' });
        let width = 0;
        for (const segment of segmenter.segment(plain)) {
            // Simple heuristic: most chars are width 1.
            // Full-width characters (CJK, emojis, etc.) are 2.
            const code = segment.segment.codePointAt(0);
            if (code && (
                (code >= 0x1100 && code <= 0x115F) || // Hangul Jamo
                (code >= 0x2329 && code <= 0x232A) || // Angle brackets
                (code >= 0x2E80 && code <= 0x303E) || // CJK Radicals / Symbols
                (code >= 0x3040 && code <= 0xA4CF) || // Hiragana / Katakana / Bopomofo / CJK Unified Ideographs
                (code >= 0xAC00 && code <= 0xD7A3) || // Hangul Syllables
                (code >= 0xF900 && code <= 0xFAFF) || // CJK Compatibility Ideographs
                (code >= 0xFE10 && code <= 0xFE19) || // Vertical forms
                (code >= 0xFE30 && code <= 0xFE6F) || // CJK Compatibility Forms
                (code >= 0xFF00 && code <= 0xFF60) || // Full-width forms
                (code >= 0xFFE0 && code <= 0xFFE6) || // Full-width currency / symbols
                code > 0x10000                        // Supplementary planes (mostly emojis)
            )) {
                width += 2;
            } else {
                width += 1;
            }
        }
        return width;
    }

    // Fallback to simple length
    return plain.length;
}

/**
 * Alias for external callers.
 */
export { stringWidth as calculateStringWidth };
