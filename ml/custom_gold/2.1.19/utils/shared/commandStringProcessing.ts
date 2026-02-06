/**
 * File: src/utils/shared/commandStringProcessing.ts
 * Role: Unified logic for parsing, tokenizing, and validating shell commands.
 */

import { randomBytes } from 'node:crypto';
import { parse as parseShellQuote, quote as quoteShellWord } from 'shell-quote';

const HEREDOC_OPERATOR_REGEX = /(?<!<)<<(?!<)(-)?(['"])?\\?(\w+)\2?/;
const HEREDOC_PREFIX = "__HEREDOC_";
const HEREDOC_SUFFIX = "__";

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

interface HeredocInfo {
    fullText: string;
    delimiter: string;
    operatorStartIndex: number;
    operatorEndIndex: number;
    contentStartIndex: number;
    contentEndIndex: number;
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
 * Tokenizes a command string into individual commands based on shell operators,
 * correctly handling heredocs.
 */
export function tokenizeCommand(commandString: string): string[] {
    const placeholders = getQuotePlaceholders();
    const { processedCommand, heredocs } = extractHeredocs(commandString);

    // Normalize backslash-newlines and apply placeholders for special characters
    const normalizedCommand = processedCommand.replace(/\\+\n/g, match => {
        const backslashes = match.length - 1;
        return backslashes % 2 === 1 ? "\\".repeat(backslashes - 1) : match;
    });

    const quoteProcessedCommand = normalizedCommand
        .replaceAll('"', `"${placeholders.DOUBLE_QUOTE}`)
        .replaceAll("'", `'${placeholders.SINGLE_QUOTE}`)
        .replaceAll("\n", `\n${placeholders.NEW_LINE}\n`)
        .replaceAll("\\(", placeholders.ESCAPED_OPEN_PAREN)
        .replaceAll("\\)", placeholders.ESCAPED_CLOSE_PAREN);

    const result = parseShellString(quoteProcessedCommand, v => `$${v}`);
    if (!result.success) return [commandString];

    const tokens = result.tokens;
    if (tokens.length === 0) return [];

    const commands: ShellToken[][] = [];
    let current: ShellToken[] = [];

    for (const token of tokens) {
        if (typeof token === "string") {
            if (token === placeholders.NEW_LINE) {
                if (current.length > 0) {
                    commands.push(current);
                    current = [];
                }
                continue;
            }
        } else if (typeof token === "object" && "op" in token) {
            if (["&&", "||", ";", "|"].includes(token.op)) {
                if (current.length > 0) {
                    commands.push(current);
                    current = [];
                }
                commands.push([token]);
                continue;
            }
        }
        current.push(token);
    }
    if (current.length > 0) commands.push(current);

    // Filter and reconstruct
    const commandStrings = commands.map(cmdTokens => {
        // Restore placeholders in each token
        const restoredTokens = cmdTokens.map(token => {
            if (typeof token === "string") {
                return token
                    .replaceAll(placeholders.SINGLE_QUOTE, "'")
                    .replaceAll(placeholders.DOUBLE_QUOTE, '"')
                    .replaceAll(placeholders.NEW_LINE, "\n")
                    .replaceAll(placeholders.ESCAPED_OPEN_PAREN, "\\(")
                    .replaceAll(placeholders.ESCAPED_CLOSE_PAREN, "\\)");
            }
            if (typeof token === "object" && "op" in token && token.op === "glob" && "pattern" in token) {
                return {
                    ...token,
                    pattern: (token as any).pattern
                        .replaceAll(placeholders.SINGLE_QUOTE, "'")
                        .replaceAll(placeholders.DOUBLE_QUOTE, '"')
                        .replaceAll(placeholders.NEW_LINE, "\n")
                        .replaceAll(placeholders.ESCAPED_OPEN_PAREN, "\\(")
                        .replaceAll(placeholders.ESCAPED_CLOSE_PAREN, "\\)")
                };
            }
            return token;
        });

        return reconstructCommand(restoredTokens, commandString);
    });

    // Final restoration of heredocs
    return commandStrings.filter(s => s.trim().length > 0).map(cmd => restoreHeredocs(cmd, heredocs));
}


/**
 * Checks if the current character at index is inside quotes.
 * Equivalent to `lB2` in chunk1458.
 */
function isInsideQuotes(command: string, index: number): boolean {
    let singleQuoted = false;
    let doubleQuoted = false;
    for (let i = 0; i < index; i++) {
        const char = command[i];
        let backslashes = 0;
        for (let j = i - 1; j >= 0 && command[j] === '\\'; j--) {
            backslashes++;
        }
        if (backslashes % 2 === 1) continue;

        if (char === "'" && !doubleQuoted) {
            singleQuoted = !singleQuoted;
        } else if (char === '"' && !singleQuoted) {
            doubleQuoted = !doubleQuoted;
        }
    }
    return singleQuoted || doubleQuoted;
}

/**
 * Checks if the character at the given index is inside a shell comment.
 * Equivalent to `iB2` in chunk1458.
 */
function isInsideComment(str: string, index: number): boolean {
    const lineStart = str.lastIndexOf("\n", index - 1) + 1;
    let singleQuoted = false;
    let doubleQuoted = false;
    for (let i = lineStart; i < index; i++) {
        const char = str[i];
        let backslashes = 0;
        for (let j = i - 1; j >= lineStart && str[j] === "\\"; j--) {
            backslashes++;
        }
        if (backslashes % 2 === 1) continue;

        if (char === "'" && !doubleQuoted) {
            singleQuoted = !singleQuoted;
        } else if (char === '"' && !singleQuoted) {
            doubleQuoted = !doubleQuoted;
        } else if (char === "#" && !singleQuoted && !doubleQuoted) {
            return true;
        }
    }
    return false;
}

/**
 * Safely appends a string to another with a space if needed.
 * Equivalent to `nt` or `clearConversation_26`.
 */
function appendWithSpace(current: string, next: string, noSpace: boolean = false): string {
    if (!current || noSpace) return current + next;
    return current + " " + next;
}

/**
 * Checks if a token is a specific operator.
 * Equivalent to `WX`.
 */
function isOperator(token: ShellToken | undefined, op: string): boolean {
    if (typeof token === 'string') return token === op;
    if (token && typeof token === 'object' && 'op' in token) return token.op === op;
    return false;
}

/**
 * Checks if a string needs quoting.
 * Equivalent to `Ku2`.
 */
export function needsQuoting(s: string): boolean {
    if (typeof s !== "string") return false;
    return /[ \t\n'"\\$;]/.test(s);
}

/**
 * Checks if the current parenthesis is the start of a subcommand.
 * Equivalent to `hDK`.
 */
function isSubcommandStart(prev: ShellToken | undefined): boolean {
    if (prev === '$') return true;
    if (isOperator(prev, '<(')) return true;
    return false;
}

/**
 * Extracts heredocs from a command string and replaces them with placeholders.
 * Robust implementation that handles multiple heredocs on the same line and <<- support.
 */
export function extractHeredocs(command: string): { processedCommand: string, heredocs: Map<string, HeredocInfo> } {
    const heredocs = new Map<string, HeredocInfo>();
    if (!command.includes("<<")) {
        return { processedCommand: command, heredocs };
    }

    const regex = new RegExp(HEREDOC_OPERATOR_REGEX.source, "g");
    const matches: HeredocInfo[] = [];
    let match;

    while ((match = regex.exec(command)) !== null) {
        const index = match.index;
        if (isInsideQuotes(command, index) || isInsideComment(command, index)) {
            continue;
        }

        const fullOpMatch = match[0];
        const isDash = match[1] === "-";
        const delimiter = match[3];
        const opEndIndex = index + fullOpMatch.length;

        // Find the start of the next line AFTER the line containing the heredoc operator(s)
        const nextNewline = command.indexOf("\n", opEndIndex);
        if (nextNewline === -1) continue;

        // Heredocs are processed line by line. Multiple heredocs on one line 
        // are processed in the order they appear, stacked vertically.
        const lineStart = command.lastIndexOf("\n", index) + 1;
        const previousMatchesOnSameLine = matches.filter(m => m.operatorStartIndex >= lineStart && m.operatorStartIndex < index);

        let contentStartIndex = nextNewline;
        if (previousMatchesOnSameLine.length > 0) {
            // Find the match that has the furthest contentEndIndex
            const sorted = [...previousMatchesOnSameLine].sort((a, b) => b.contentEndIndex - a.contentEndIndex);
            contentStartIndex = sorted[0].contentEndIndex;
        }

        const afterContentStart = command.slice(contentStartIndex);
        // Important: we split by \n but need to keep the \n in the content eventually
        let delimiterLineIndex = -1;

        // Skip the very first empty string if contentStartIndex was a newline
        const searchOffset = afterContentStart.startsWith("\n") ? 1 : 0;
        const searchableAfterContentStart = afterContentStart.slice(searchOffset);
        const searchableLines = searchableAfterContentStart.split("\n");

        for (let i = 0; i < searchableLines.length; i++) {
            const line = searchableLines[i];
            const trimmedLine = isDash ? line.trimStart() : line;
            if (trimmedLine === delimiter) {
                delimiterLineIndex = i;
                break;
            }
        }

        if (delimiterLineIndex === -1) continue;

        // Reconstruct content including all newlines
        const contentLines = searchableLines.slice(0, delimiterLineIndex + 1);
        const contentText = contentLines.join("\n");
        const actualContentStartIndex = contentStartIndex + searchOffset;
        const contentEndIndex = actualContentStartIndex + contentText.length;

        matches.push({
            // fullText includes the operator and the content
            fullText: fullOpMatch + "\n" + contentText,
            delimiter,
            operatorStartIndex: index,
            operatorEndIndex: opEndIndex,
            contentStartIndex: actualContentStartIndex - 1, // include the preceding newline
            contentEndIndex
        });
    }

    if (matches.length === 0) {
        return { processedCommand: command, heredocs };
    }

    // Sort by contentEndIndex descending to avoid displacement during slicing
    const sortedMatches = [...matches].sort((a, b) => b.contentEndIndex - a.contentEndIndex);

    const nonce = randomBytes(8).toString("hex");
    let processed = command;

    sortedMatches.forEach((m, i) => {
        const placeholder = `${HEREDOC_PREFIX}${sortedMatches.length - 1 - i}_${nonce}${HEREDOC_SUFFIX}`;
        heredocs.set(placeholder, m);
        // Remove the content (including its preceding newline)
        processed = processed.slice(0, m.contentStartIndex) + processed.slice(m.contentEndIndex);
        // Then replace the operator with the placeholder
        processed = processed.slice(0, m.operatorStartIndex) + placeholder + processed.slice(m.operatorEndIndex);
    });

    return { processedCommand: processed, heredocs };
}


/**
 * Restores heredocs from placeholders in a command string.
 * Equivalent to `nB2` in chunk1458.
 */
export function restoreHeredocs(command: string, heredocs: Map<string, HeredocInfo>): string {
    let result = command;
    for (const [placeholder, info] of heredocs) {
        result = result.replaceAll(placeholder, info.fullText);
    }
    return result;
}

/**
 * Reconstructs the command from its parsed tokens.
 */
export function reconstructCommand(tokens: ShellToken[], originalCommand: string): string {
    if (!tokens.length) {
        return originalCommand;
    }

    let reconstructed = "";
    let parenDepth = 0;
    let insideProcessSubstitution = false;

    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const prev = tokens[i - 1];
        const next = tokens[i + 1];

        if (typeof token === 'string') {
            // Check if it needs quoting
            let quoted = /[|&;]/.test(token) ? `"${token}"` : needsQuoting(token) ? quoteShellWord([token]) : token;

            // Handle process substitution or special join rules
            // W = reconstructed ends with "(" OR prev is "$" OR prev is ")" operator
            const noSpaceBefore = reconstructed.endsWith("(") || prev === "$" || isOperator(prev, ")");

            if (reconstructed.endsWith("<(")) {
                reconstructed += " " + quoted;
            } else {
                reconstructed = appendWithSpace(reconstructed, quoted, noSpaceBefore);
            }
            continue;
        }

        if (typeof token !== 'object' || !token || !('op' in token)) {
            continue;
        }

        const op = token.op;

        if (op === 'glob' && 'pattern' in token) {
            reconstructed = appendWithSpace(reconstructed, (token as any).pattern);
            continue;
        }

        // Handle numeric redirects like 2>&1
        if (op === ">&" && typeof prev === "string" && /^\d+$/.test(prev) && typeof next === "string" && /^\d+$/.test(next)) {
            const lastIndex = reconstructed.lastIndexOf(prev);
            reconstructed = reconstructed.slice(0, lastIndex) + prev + op + next;
            i++;
            continue;
        }

        // Handle heredocs
        if (op === "<" && isOperator(next, "<")) {
            const nextNext = tokens[i + 2];
            if (nextNext && typeof nextNext === "string") {
                reconstructed = appendWithSpace(reconstructed, nextNext);
                i += 2;
                continue;
            }
        }

        if (op === "<<<") {
            reconstructed = appendWithSpace(reconstructed, op);
            continue;
        }

        if (op === "(") {
            if (isSubcommandStart(prev) || parenDepth > 0) {
                parenDepth++;
                if (reconstructed.endsWith(" ")) {
                    reconstructed = reconstructed.slice(0, -1);
                }
                reconstructed += "(";
            } else {
                if (reconstructed.endsWith("$")) {
                    parenDepth++;
                    reconstructed += "(";
                } else {
                    const noSpace = reconstructed.endsWith("<(") || reconstructed.endsWith("(");
                    reconstructed = appendWithSpace(reconstructed, "(", noSpace);
                }
            }
            continue;
        }

        if (op === ")") {
            if (insideProcessSubstitution) {
                insideProcessSubstitution = false;
                reconstructed += ")";
                continue;
            }
            if (parenDepth > 0) {
                parenDepth--;
            }
            reconstructed += ")";
            continue;
        }

        if (op === "<(") {
            insideProcessSubstitution = true;
            reconstructed = appendWithSpace(reconstructed, op);
            continue;
        }

        if (["&&", "||", "|", ";", ">", ">>", "<"].includes(op)) {
            reconstructed = appendWithSpace(reconstructed, op);
        }
    }

    return reconstructed.trim() || originalCommand;
}

/**
 * Checks if a command string is safe to execute.
 */
export function isCommandSafe(commandString: string): boolean {
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
