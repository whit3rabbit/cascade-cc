import { parse, Token, quote } from "../../vendor/shell-quote.js";
import { randomBytes } from "node:crypto";

// Logic from chunk_581.ts (Bash Command Parsing & Redirection)

const KM0 = "__SINGLE_QUOTE__";
const VM0 = "__DOUBLE_QUOTE__";
const IM0 = "__NEW_LINE__";
const vW9 = "__ESCAPED_OPEN_PAREN__";
const kW9 = "__ESCAPED_CLOSE_PAREN__";

const sY7 = "__HEREDOC_";
const tY7 = "__";
const AJ7 = /(?<!<)<<(?!<)(-)?(['"])?\\?(\w+)\2?/;

const KfA = new Set(["0", "1", "2"]);
const hW9 = new Set(["&&", "||", ";", ";;", "|"]);
const XJ7 = new Set([...hW9, ">&", ">", ">>"]);

export interface HeredocInfo {
    fullText: string;
    delimiter: string;
    operatorStartIndex: number;
    operatorEndIndex: number;
    contentStartIndex: number;
    contentEndIndex: number;
}

/**
 * Checks if a command is a simple help request. (JJ7)
 */
export function isHelpCommand(command: string): boolean {
    const trimmed = command.trim();
    if (!trimmed.endsWith("--help")) return false;
    if (trimmed.includes('"') || trimmed.includes("'")) return false;

    const result = tokenizeCommand(trimmed);
    if (!result || result.length === 0) return false;

    let hasHelp = false;
    for (const token of result) {
        if (typeof token === "string") {
            if (token.startsWith("-")) {
                if (token === "--help") hasHelp = true;
                else return false;
            } else if (!/^[a-zA-Z0-9.\-/]+$/.test(token)) {
                return false;
            }
        }
    }
    return hasHelp;
}

/**
 * Tokenizes a command using the deobfuscated shell-quote logic. (KE0)
 */
export function tokenizeCommand(command: string): Token[] {
    try {
        const { processedCommand, heredocs } = extractHeredocs(command);

        // Pre-process for special characters
        const preprocessed = processedCommand
            .replaceAll('"', `"${VM0}`)
            .replaceAll("'", `'${KM0}`)
            .replaceAll("\n", `\n${IM0}\n`)
            .replaceAll("\\(", vW9)
            .replaceAll("\\)", kW9);

        const result = parse(preprocessed, (key) => `$${key}`);
        if (!result) return [];

        const tokens: Token[] = [];
        for (const token of result) {
            if (typeof token === "string") {
                if (tokens.length > 0 && typeof tokens[tokens.length - 1] === "string") {
                    if (token === IM0) tokens.push("" as any); // Marker for newline join logic
                    else (tokens[tokens.length - 1] as string) += " " + token;
                    continue;
                }
            } else if (typeof token === "object" && "op" in token && token.op === "glob") {
                if (tokens.length > 0 && typeof tokens[tokens.length - 1] === "string") {
                    (tokens[tokens.length - 1] as string) += " " + (token as any).pattern;
                    continue;
                }
            }
            tokens.push(token);
        }

        // Post-process tokens (clean markers and restore heredocs)
        return tokens.map(token => {
            if (typeof token === "string") {
                let cleaned = token
                    .replaceAll(KM0, "'")
                    .replaceAll(VM0, '"')
                    .replaceAll(`\n${IM0}\n`, "\n")
                    .replaceAll(vW9, "\\(")
                    .replaceAll(kW9, "\\)");

                // Restore heredocs
                for (const [key, info] of heredocs) {
                    cleaned = cleaned.replaceAll(key, info.fullText);
                }
                return cleaned;
            }
            // If it's an operator object, we just return it
            return token;
        }).filter(t => t !== null && t !== "");
    } catch (err) {
        return [command];
    }
}

/**
 * Strips redirections from a bash command and returns the extracted redirections. (FS)
 */
export function stripRedirections(command: string): { commandWithoutRedirections: string; redirections: any[] } {
    const redirections: any[] = [];
    const tokens = parse(command, (key) => `$${key}`);
    if (!tokens) return { commandWithoutRedirections: command, redirections: [] };

    const scriptStarts = new Set<number>();
    const parenStack: { index: number, isStart: boolean }[] = [];

    // Identify significant parentheses (D1A logic)
    tokens.forEach((token, i) => {
        if (isOp(token, "(")) {
            const prev = tokens[i - 1];
            const isStart = !prev || (typeof prev === "object" && "op" in prev && ["&&", "||", ";", "|"].includes(prev.op));
            parenStack.push({ index: i, isStart: !!isStart });
        } else if (isOp(token, ")") && parenStack.length > 0) {
            const start = parenStack.pop()!;
            const next = tokens[i + 1];
            if (start.isStart && (isOp(next, ">") || isOp(next, ">>"))) {
                scriptStarts.add(start.index).add(i);
            }
        }
    });

    const resultTokens: Token[] = [];
    let subshellDepth = 0;

    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const prev = tokens[i - 1];
        const next = tokens[i + 1];
        const nextNext = tokens[i + 2];
        const next3 = tokens[i + 3];

        if ((isOp(token, "(") || isOp(token, ")")) && scriptStarts.has(i)) continue;

        if (isOp(token, "(") && typeof prev === "string" && prev.endsWith("$")) {
            subshellDepth++;
        } else if (isOp(token, ")") && subshellDepth > 0) {
            subshellDepth--;
        }

        if (subshellDepth === 0) {
            const skipResult = processRedirection(token, prev, next, nextNext, next3, redirections, resultTokens);
            if (skipResult.skip > 0) {
                i += skipResult.skip;
                continue;
            }
        }
        resultTokens.push(token);
    }

    return {
        commandWithoutRedirections: reassembleCommand(resultTokens, command),
        redirections
    };
}

function processRedirection(token: Token, prev: Token, next: Token, nextNext: Token, nextNextNext: Token, redirections: any[], resultTokens: Token[]): { skip: number } {
    const isDigit = (t: Token): t is string => typeof t === "string" && /^\d+$/.test(t.trim());
    const isStandard = (t: Token) => typeof t === "string" && !t.startsWith("!") && !t.startsWith("~") && !t.includes("$") && !t.includes("`") && !t.includes("*") && !t.includes("?");

    if (isOp(token, ">") || isOp(token, ">>")) {
        const op = (token as any).op;
        if (isDigit(prev)) {
            if (next === "!" && isStandard(nextNext)) return extractRedir(prev.trim(), op, nextNext, redirections, resultTokens, 2);
            if (isOp(next, "|") && isStandard(nextNext)) return extractRedir(prev.trim(), op, nextNext, redirections, resultTokens, 2);
            return extractRedir(prev.trim(), op, next, redirections, resultTokens, 1);
        }
        if (isOp(next, "|") && isStandard(nextNext)) {
            redirections.push({ target: nextNext, operator: op });
            return { skip: 2 };
        }
        if (next === "!" && isStandard(nextNext)) {
            redirections.push({ target: nextNext, operator: op });
            return { skip: 2 };
        }
        if (isStandard(next)) {
            redirections.push({ target: next, operator: op });
            return { skip: 1 };
        }
    }

    if (isOp(token, ">&")) {
        if (isDigit(prev) && isDigit(next)) return { skip: 0 };
        if (isOp(next, "|") && isStandard(nextNext)) {
            redirections.push({ target: nextNext, operator: ">" });
            return { skip: 2 };
        }
        if (next === "!" && isStandard(nextNext)) {
            redirections.push({ target: nextNext, operator: ">" });
            return { skip: 2 };
        }
        if (isStandard(next) && !isDigit(next)) {
            redirections.push({ target: next, operator: ">" });
            return { skip: 1 };
        }
    }

    return { skip: 0 };
}

function extractRedir(descriptor: string, op: string, target: Token, redirections: any[], resultTokens: Token[], skip: number) {
    const isStandardOut = descriptor === "1";
    const isLiteralTarget = target && typeof target === "string" && !/^\d+$/.test(target);

    if (resultTokens.length > 0) resultTokens.pop(); // Remove the descriptor digit

    if (isLiteralTarget) {
        redirections.push({ target, operator: op });
        if (!isStandardOut) resultTokens.push(descriptor + op, target);
        return { skip };
    }

    if (!isStandardOut) {
        resultTokens.push(descriptor + op);
        if (target) {
            resultTokens.push(target);
            return { skip: 1 };
        }
    }
    return { skip: 0 };
}

/**
 * Checks if a command is "safe" or "standard" (no complex redirections or subshells). (IJ7)
 */
export function isSandboxedSafe(command: string): boolean {
    const { processedCommand } = extractHeredocs(command);
    const result = parse(processedCommand.replaceAll('"', `"${VM0}`).replaceAll("'", `'${KM0}`), (key) => `$${key}`);
    if (!result) return false;

    for (let i = 0; i < result.length; i++) {
        const token = result[i];
        const next = result[i + 1];

        if (!token) continue;
        if (typeof token === "string") continue;
        if ("comment" in token) return false;
        if ("op" in token) {
            const op = token.op;
            if (op === "glob") continue;
            if (hW9.has(op)) continue;
            if (op === ">&") {
                if (next && typeof next === "string" && KfA.has(next.trim())) continue;
            } else if (op === ">" || op === ">>") {
                continue;
            }
            return false;
        }
    }
    return true;
}

/**
 * Normalizes and reassembles a command from tokens. (VJ7)
 */
export function reassembleCommand(tokens: Token[], original: string): string {
    if (!tokens.length) return original;
    let command = "";
    let subshellStack = 0;
    let inProcessSubstitution = false;

    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const prev = tokens[i - 1];
        const next = tokens[i + 1];

        if (typeof token === "string") {
            const quoted = /[|&;]/.test(token) ? `"${token}"` : (token.includes(" ") || token.includes("\t") ? quote([token]) : token);
            const isVarPrefix = quoted.endsWith("$");
            const isSubshellNext = next && typeof next === "object" && "op" in next && next.op === "(";
            const isWrapped = command.endsWith("(") || prev === "$" || (typeof prev === "object" && "op" in prev && prev.op === ")");

            if (command.endsWith("<(")) command += " " + quoted;
            else command = joinWithSpace(command, quoted, isWrapped);
            continue;
        }

        if (typeof token !== "object" || !token || !("op" in token)) continue;

        const op = token.op;
        if (op === "glob" && "pattern" in token) {
            command = joinWithSpace(command, token.pattern);
            continue;
        }

        if (op === ">&" && typeof prev === "string" && /^\d+$/.test(prev) && typeof next === "string" && /^\d+$/.test(next)) {
            const lastIndex = command.lastIndexOf(prev);
            command = command.slice(0, lastIndex) + prev + op + next;
            i++;
            continue;
        }

        if (op === "<" && isOp(next, "<")) {
            const nextNext = tokens[i + 2];
            if (nextNext && typeof nextNext === "string") {
                command = joinWithSpace(command, nextNext);
                i += 2;
                continue;
            }
        }

        if (op === "<<<") {
            command = joinWithSpace(command, op);
            continue;
        }

        if (op === "(") {
            if (isSubshellVar(prev, tokens, i) || subshellStack > 0) {
                subshellStack++;
                if (command.endsWith(" ")) command = command.slice(0, -1);
                command += "(";
            } else if (command.endsWith("$")) {
                if (isSubshellVar(prev, tokens, i)) {
                    subshellStack++;
                    command += "(";
                } else command = joinWithSpace(command, "(");
            } else {
                const noSpace = command.endsWith("<(") || command.endsWith("(");
                command = joinWithSpace(command, "(", noSpace);
            }
            continue;
        }

        if (op === ")") {
            if (inProcessSubstitution) {
                inProcessSubstitution = false;
                command += ")";
                continue;
            }
            if (subshellStack > 0) subshellStack--;
            command += ")";
            continue;
        }

        if (op === "<(") {
            inProcessSubstitution = true;
            command = joinWithSpace(command, op);
            continue;
        }

        if (["&&", "||", "|", ";", ">", ">>", "<"].includes(op)) {
            command = joinWithSpace(command, op);
        }
    }

    return command.trim() || original;
}

function joinWithSpace(base: string, next: string, noSpace: boolean = false): string {
    if (!base || noSpace) return base + next;
    return base + " " + next;
}

function isSubshellVar(prev: Token, tokens: Token[], index: number): boolean {
    if (!prev || typeof prev !== "string") return false;
    if (prev === "$") return true;
    if (prev.endsWith("$")) {
        if (prev.includes("=") && prev.endsWith("=$")) return true;
        let depth = 1;
        for (let j = index + 1; j < tokens.length && depth > 0; j++) {
            if (isOp(tokens[j], "(")) depth++;
            if (isOp(tokens[j], ")") && --depth === 0) {
                const after = tokens[j + 1];
                return !!(after && typeof after === "string" && !after.startsWith(" "));
            }
        }
    }
    return false;
}

function isOp(token: Token, opString: string): boolean {
    return typeof token === "object" && token !== null && "op" in token && token.op === opString;
}

/**
 * Extracts heredocs and replaces them with markers. (XM0)
 */
export function extractHeredocs(command: string): { processedCommand: string; heredocs: Map<string, HeredocInfo> } {
    const heredocs = new Map<string, HeredocInfo>();
    if (!command.includes("<<")) return { processedCommand: command, heredocs };

    const regex = new RegExp(AJ7.source, "g");
    const found: HeredocInfo[] = [];
    let match;

    while ((match = regex.exec(command)) !== null) {
        const index = match.index;
        if (isInsideQuotes(command, index)) continue;
        if (isInsideComment(command, index)) continue;

        const fullMatch = match[0];
        const delimiter = match[3];
        const afterOperator = index + fullMatch.length;
        const newlineIndex = command.slice(afterOperator).indexOf("\n");

        if (newlineIndex === -1) continue;

        const contentStart = afterOperator + newlineIndex;
        const lines = command.slice(contentStart + 1).split("\n");
        let delimiterLineIndex = -1;

        for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim() === delimiter) {
                delimiterLineIndex = i;
                break;
            }
        }

        if (delimiterLineIndex === -1) continue;

        const contentLines = lines.slice(0, delimiterLineIndex + 1);
        const contentLength = contentLines.join("\n").length;
        const contentEnd = contentStart + 1 + contentLength;

        found.push({
            fullText: command.slice(index, contentEnd),
            delimiter,
            operatorStartIndex: index,
            operatorEndIndex: afterOperator,
            contentStartIndex: contentStart,
            contentEndIndex: contentEnd
        });
    }

    if (found.length === 0) return { processedCommand: command, heredocs };

    // Filter nested (not common in this usage but deobfuscated logic did this)
    const filtered = found.filter((h, _, all) => {
        for (const other of all) {
            if (h === other) continue;
            if (h.operatorStartIndex > other.contentStartIndex && h.operatorStartIndex < other.contentEndIndex) return false;
        }
        return true;
    });

    if (filtered.length === 0) return { processedCommand: command, heredocs };

    // Avoid duplicates on same indexes
    if (new Set(filtered.map(h => h.contentStartIndex)).size < filtered.length) return { processedCommand: command, heredocs };

    filtered.sort((a, b) => b.contentEndIndex - a.contentEndIndex);
    const heredocId = randomBytes(4).toString("hex");
    let processed = command;

    filtered.forEach((h, i) => {
        const marker = `${sY7}${i}_${heredocId}${tY7}`;
        heredocs.set(marker, h);
        processed = processed.slice(0, h.operatorStartIndex) + marker + processed.slice(h.operatorEndIndex, h.contentStartIndex) + processed.slice(h.contentEndIndex);
    });

    return { processedCommand: processed, heredocs };
}

function isInsideQuotes(text: string, pos: number): boolean {
    let single = false;
    let double = false;
    for (let i = 0; i < pos; i++) {
        const char = text[i];
        let escapeCount = 0;
        for (let j = i - 1; j >= 0 && text[j] === "\\"; j--) escapeCount++;
        if (escapeCount % 2 === 1) continue;

        if (char === "'" && !double) single = !single;
        else if (char === '"' && !single) double = !double;
    }
    return single || double;
}

function isInsideComment(text: string, pos: number): boolean {
    const lineStart = text.lastIndexOf("\n", pos - 1) + 1;
    let single = false;
    let double = false;
    for (let i = lineStart; i < pos; i++) {
        const char = text[i];
        let escapeCount = 0;
        for (let j = i - 1; j >= lineStart && text[j] === "\\"; j--) escapeCount++;
        if (escapeCount % 2 === 1) continue;

        if (char === "'" && !double) single = !single;
        else if (char === '"' && !single) double = !double;
        else if (char === "#" && !single && !double) return true;
    }
    return false;
}
