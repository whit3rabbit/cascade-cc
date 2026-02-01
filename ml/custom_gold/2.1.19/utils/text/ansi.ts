/**
 * File: src/utils/text/ansi.ts
 * Role: Utilities for handling ANSI escape codes.
 */

export const ANSI_REGEX = /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;

/**
 * Strips ANSI escape codes from a string.
 * 
 * @param {string} text - The string to strip.
 * @returns {string} The stripped string.
 */
export function stripAnsi(text: string): string {
    if (typeof text !== 'string') {
        return text;
    }
    return text.replace(ANSI_REGEX, '');
}

/**
 * Alias for legacy/obfuscated usage and compatibility.
 */
export const eU = stripAnsi;
export const AH = stripAnsi;

export const truncateString = (args: string[]): string => {
    // Simple mock since we don't have the original truncate logic here
    return args.join(' ').substring(0, 1000);
};

export default stripAnsi;
