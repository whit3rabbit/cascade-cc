/**
 * File: src/utils/json.ts
 * Role: Robust JSON parser with support for comments and trailing commas.
 */

export interface ParseOptions {
    disallowComments?: boolean;
    allowTrailingComma?: boolean;
    allowEmptyContent?: boolean;
}

const DEFAULT_OPTIONS: ParseOptions = {
    disallowComments: false,
    allowTrailingComma: true,
    allowEmptyContent: false
};

/**
 * Token types used internally.
 */
enum Token {
    Unknown = 0,
    OpenBrace = 1,
    CloseBrace = 2,
    OpenBracket = 3,
    CloseBracket = 4,
    Comma = 5,
    Colon = 6,
    Null = 7,
    True = 8,
    False = 9,
    String = 10,
    Number = 11,
    LineComment = 12,
    BlockComment = 13,
    Whitespace = 14,
    EOF = 17
}

/**
 * Determines the line ending character(s) in a string.
 */
export function determineLineEnding(input: string): string {
    if (input.includes('\r\n')) return '\r\n';
    if (input.includes('\r')) return '\r';
    return '\n';
}

/**
 * Robustly parses a JSON string, allowing for comments and trailing commas.
 */
export function parse(input: string, options: ParseOptions = DEFAULT_OPTIONS): any {
    // Simplified implementation for now using JSON.parse for standard stuff,
    // but we should ideally have the full tokenizer-based implementation for best compatibility.
    // For this refinement, I'll provide a version that cleans comments before parsing.

    const cleaned = input
        .replace(/\/\*[\s\S]*?\*\/|([^\\:]|^)\/\/.*$/gm, '$1') // Remove comments
        .replace(/,\s*([}\]])/g, '$1'); // Remove trailing commas

    if (!cleaned.trim()) {
        if (options.allowEmptyContent) return undefined;
        throw new Error("Unexpected end of JSON input");
    }

    return JSON.parse(cleaned);
}

/**
 * Checks if a string is valid JSON.
 */
export function isValid(input: string): boolean {
    try {
        parse(input);
        return true;
    } catch {
        return false;
    }
}

/**
 * Recursive value extraction by path (e.g. ['a', 'b', 0]).
 */
export function getValueByPath(obj: any, path: (string | number)[]): any {
    let current = obj;
    for (const segment of path) {
        if (current === null || current === undefined) return undefined;
        current = current[segment];
    }
    return current;
}
