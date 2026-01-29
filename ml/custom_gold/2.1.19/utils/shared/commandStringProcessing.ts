/**
 * File: src/utils/shared/commandStringProcessing.ts
 * Role: Unified logic for parsing, tokenizing, and validating shell commands.
 */

import { randomBytes } from 'node:crypto';
import { parse as parseShellQuote, quote as quoteShellWord } from 'shell-quote';

// --- Types ---
export interface QuotePlaceholders {
    SINGLE_QUOTE: string;
    DOUBLE_QUOTE: string;
    NEW_LINE: string;
    ESCAPED_OPEN_PAREN: string;
    ESCAPED_CLOSE_PAREN: string;
}

export type ShellToken = string | { op: string } | { comment: string } | { glob: string, pattern: string };

export interface Redirection {
    target: string;
    operator: string;
}

export interface CommandParseResult {
    commandWithoutRedirections: string;
    redirections: string[];
    hasDangerousRedirection: boolean;
}

// --- Constants ---
const CD_COMMAND_REGEX = /^cd(?:\s|$)/;
const REDIRECTION_NUMBERS = new Set(["0", "1", "2"]);
const SHELL_OPERATORS = new Set(["&&", "||", ";", ";;", "|"]);

/**
 * Generates random strings to use as placeholders for shell-related characters.
 */
export function getQuotePlaceholders(): QuotePlaceholders {
    const nonce = randomBytes(8).toString("hex");
    return {
        SINGLE_QUOTE: `__SINGLE_QUOTE_${nonce}__`,
        DOUBLE_QUOTE: `__DOUBLE_QUOTE_${nonce}__`,
        NEW_LINE: `__NEW_LINE_${nonce}__`,
        ESCAPED_OPEN_PAREN: `__ESCAPED_OPEN_PAREN_${nonce}__`,
        ESCAPED_CLOSE_PAREN: `__ESCAPED_CLOSE_PAREN_${nonce}__`
    };
}

/**
 * Checks if a word is "safe" (contains no shell special characters).
 */
export function isSafeWord(str: any): boolean {
    if (typeof str !== "string") return false;
    return !str.startsWith("!") &&
        !str.includes("$") &&
        !str.includes("`") &&
        !str.includes("*") &&
        !str.includes("?") &&
        !str.includes("[") &&
        !str.includes("{") &&
        !str.includes("~") &&
        !str.includes("(") &&
        !str.includes("<") &&
        !str.startsWith("&");
}

/**
 * Checks if a string contains variable expansion characters ($ or %).
 */
export function containsVariableExpansion(str: any): boolean {
    return typeof str === "string" && (str.includes("$") || str.includes("%"));
}

/**
 * Wrapper for shell-quote.parse with error handling.
 */
export function parseShellString(command: string, env?: Record<string, string> | ((v: string) => string)): { success: boolean, tokens: ShellToken[], error?: string } {
    try {
        return {
            success: true,
            tokens: parseShellQuote(command, env as any) as ShellToken[]
        };
    } catch (err: any) {
        return {
            success: false,
            tokens: [],
            error: err instanceof Error ? err.message : "Parse error"
        };
    }
}

/**
 * Tokenizes a command string into individual commands based on shell operators.
 */
export function tokenizeCommand(commandString: string): string[] {
    const result = parseShellString(commandString);
    if (!result.success) return [commandString];

    const commands: string[] = [];
    let current: ShellToken[] = [];

    for (const token of result.tokens) {
        if (typeof token === 'object' && 'op' in token && ['&&', '||', ';', '|'].includes(token.op)) {
            if (current.length > 0) {
                commands.push(reconstructCommand(current, commandString));
                current = [];
            }
        } else {
            current.push(token);
        }
    }

    if (current.length > 0) {
        commands.push(reconstructCommand(current, commandString));
    }

    return commands;
}

/**
 * Checks if a string needs quoting for safe shell usage.
 */
export function needsQuoting(str: any): boolean {
    if (typeof str !== "string") return false;
    return str.includes(" ") || str.includes("\t") || str.includes("\n") || /[|&;()<>$!*?{}[\]]/.test(str);
}

/**
 * Checks if a token represents the start of a subshell or variable expansion.
 */
function isSubshellOrVariableExpansion(token: any): boolean {
    return typeof token === "object" && token && "op" in token && (token.op === "(" || token.op === "$(");
}

/**
 * Concatenates two strings with a space in between, unless already separated.
 */
function concatenateWithSpace(a: string, b: string, forceConcat = false): string {
    if (!a || forceConcat) {
        return a + b;
    }
    return a + " " + b;
}

/**
 * Reconstructs the command from its parsed tokens.
 */
export function reconstructCommand(tokens: ShellToken[], originalCommand: string): string {
    if (!tokens.length) {
        return originalCommand;
    }

    let reconstructed = "";
    let inParenthesis = 0;
    let z = false; // Flag for process substitution "<("

    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const previousToken = tokens[i - 1];
        const nextToken = tokens[i + 1];

        if (typeof token === "string") {
            const escapedToken = /[|&;]/.test(token) ? `"${token}"` : needsQuoting(token) ? quoteShellWord([token]) : token;
            const alreadyInsideParens = reconstructed.endsWith("(") || previousToken === "$" || (typeof previousToken === "object" && previousToken && "op" in previousToken && previousToken.op === ")");

            if (reconstructed.endsWith("<(")) {
                reconstructed += " " + escapedToken;
            } else {
                reconstructed = concatenateWithSpace(reconstructed, escapedToken, alreadyInsideParens);
            }
            continue;
        }

        if (typeof token !== "object" || !token || !("op" in token)) {
            continue;
        }

        const { op: operator } = token;

        if (operator === "glob" && "pattern" in token) {
            reconstructed = concatenateWithSpace(reconstructed, (token as any).pattern);
            continue;
        }


        if (operator === ">&" && typeof previousToken === "string" && /^\d+$/.test(previousToken) && typeof nextToken === "string" && /^\d+$/.test(nextToken)) {
            const lastIndex = reconstructed.lastIndexOf(previousToken);
            reconstructed = reconstructed.slice(0, lastIndex) + previousToken + operator + nextToken;
            i++;
            continue;
        }

        if (operator === "<" && typeof nextToken === 'object' && nextToken && 'op' in nextToken && nextToken.op === "<") {
            // Heredoc sequence
            // This is a simplified reconstruction for heredocs
            const nextNextToken = tokens[i + 2];
            if (nextNextToken && typeof nextNextToken === "string") {
                reconstructed = concatenateWithSpace(reconstructed, "<< " + nextNextToken);
                i += 2;
                continue;
            }
        }

        if (operator === "(") {
            if (isSubshellOrVariableExpansion(previousToken) || inParenthesis > 0) {
                inParenthesis++;
                if (reconstructed.endsWith(" ")) reconstructed = reconstructed.slice(0, -1);
                reconstructed += "(";
            } else {
                if (reconstructed.endsWith("$")) {
                    inParenthesis++;
                    reconstructed += "(";
                } else {
                    const lastWasLessThanParen = reconstructed.endsWith("<(") || reconstructed.endsWith("(");
                    reconstructed = concatenateWithSpace(reconstructed, "(", lastWasLessThanParen);
                }
            }
            continue;
        }

        if (operator === ")") {
            if (z) {
                z = false;
                reconstructed += ")";
                continue;
            }
            if (inParenthesis > 0) inParenthesis--;
            reconstructed += ")";
            continue;
        }

        if (operator === "<(") {
            z = true;
            reconstructed = concatenateWithSpace(reconstructed, operator);
            continue;
        }

        if (["&&", "||", "|", ";", ">", ">>", "<", "<<<"].includes(operator)) {
            reconstructed = concatenateWithSpace(reconstructed, operator);
        }
    }

    return reconstructed.trim() || originalCommand;
}

/**
 * Checks if a command string is safe to execute.
 */
export function isCommandSafe(commandString: string): boolean {
    const placeholders = getQuotePlaceholders();
    // Simplified safe check
    const tokens = parseShellString(commandString, (w) => `$${w}`);

    if (!tokens.success) return false;

    for (let i = 0; i < tokens.tokens.length; i++) {
        const token = tokens.tokens[i];
        const nextToken = tokens.tokens[i + 1];

        if (token === undefined) continue;
        if (typeof token === "string") continue;
        if ("comment" in token) return false;

        if ("op" in token) {
            if (token.op === "glob") continue;
            if (SHELL_OPERATORS.has(token.op)) continue;
            if (token.op === ">") continue;
            if (token.op === ">>") continue;
            if (token.op === ">&") {
                if (nextToken !== undefined && typeof nextToken === "string" && REDIRECTION_NUMBERS.has(nextToken.trim())) {
                    continue;
                }
            }
            return false;
        }
    }
    return true;
}

/**
 * Checks if a command string contains dangerous elements.
 */
export function isCommandPotentiallyDangerous(commandString: string): boolean {
    try {
        return tokenizeCommand(commandString).length > 1 && !isCommandSafe(commandString);
    } catch {
        return true;
    }
}

/**
 * Checks if a command string contains a 'cd' command.
 */
export function containsCdCommand(commandString: string): boolean {
    return tokenizeCommand(commandString).some(part => CD_COMMAND_REGEX.test(part.trim()));
}

/**
 * Parses the command string and separates it into a command and redirections.
 */
export function parseCommandWithRedirections(commandString: string): CommandParseResult {
    const parsed = parseShellString(commandString, token => `$${token}`);
    if (!parsed.success) {
        return {
            commandWithoutRedirections: commandString,
            redirections: [],
            hasDangerousRedirection: false
        };
    }

    const tokens = parsed.tokens;
    const commandParts: ShellToken[] = [];
    const redirections: string[] = [];
    let hasDangerousRedirection = false;

    // Redirection parsing logic is complex; this is a simplified robust version
    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        if (typeof token === 'object' && 'op' in token) {
            if (['>', '>>', '>&'].includes(token.op)) {
                // Simplified: treat any redirection with variable expansion as dangerous
                const nextToken = tokens[i + 1];
                if (nextToken && typeof nextToken === 'string' && containsVariableExpansion(nextToken)) {
                    hasDangerousRedirection = true;
                }
                if (nextToken) {
                    redirections.push(token.op, nextToken.toString());
                    i++; // skip target
                    continue;
                }
            }
        }
        commandParts.push(token);
    }

    return {
        commandWithoutRedirections: reconstructCommand(commandParts, commandString),
        redirections,
        hasDangerousRedirection
    };
}

/**
 * Recursively expands environment variables in a string and identifies missing ones.
 */
export function getMissingVariables(str: string, env: Record<string, string | undefined> = process.env): { expanded: string, missingVars: string[] } {
    if (typeof str !== 'string') return { expanded: str, missingVars: [] };

    const missingVars = new Set<string>();

    // Match ${VAR} and $VAR
    const expanded = str.replace(/\$(\{([^}]+)\}|([a-zA-Z_][a-zA-Z0-9_]*))/g, (match, p1, p2, p3) => {
        const varName = (p2 || p3) as string;
        const value = env[varName];
        if (value === undefined) {
            missingVars.add(varName);
            return match;
        }
        return value;
    });

    return { expanded, missingVars: Array.from(missingVars) };
}
