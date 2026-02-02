/**
 * File: src/utils/terminal/shellQuote.ts
 * Role: Robust shell escaping for command arguments.
 */

import { quote } from 'shell-quote';

/**
 * Safely quotes strings for use in shell commands.
 * 
 * @param args - Array of strings to quote and join.
 * @returns A single string with arguments safely quoted and space-separated.
 */
export function shellQuote(args: string[]): string {
    return quote(args);
}

/**
 * Alias for compatibility with gold reference R4.
 */
export const R4 = shellQuote;
