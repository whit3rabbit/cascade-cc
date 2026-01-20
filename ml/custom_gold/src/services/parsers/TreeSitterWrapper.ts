// @ts-ignore
import { Parser, Language, Tree, Node as SyntaxNode, TreeCursor } from 'web-tree-sitter';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

// Constants
export const BASH_SECURITY_CHECKS = {
    INCOMPLETE_COMMANDS: 1,
    JQ_SYSTEM_FUNCTION: 2,
    JQ_FILE_ARGUMENTS: 3,
    OBFUSCATED_FLAGS: 4,
    SHELL_METACHARACTERS: 5,
    DANGEROUS_VARIABLES: 6,
    NEWLINES: 7,
    DANGEROUS_PATTERNS_COMMAND_SUBSTITUTION: 8,
    DANGEROUS_PATTERNS_INPUT_REDIRECTION: 9,
    DANGEROUS_PATTERNS_OUTPUT_REDIRECTION: 10,
    IFS_INJECTION: 11,
    GIT_COMMIT_SUBSTITUTION: 12,
    PROC_ENVIRON_ACCESS: 13
};

// --- Helpers identifying original obfuscated names ---
// ou5 -> checkIfsInjection
// ru5 -> checkProcEnvironAccess
// su5 -> checkObfuscatedFlags
// ku5 -> extractCommandInfo (from chunk_461)
// bu5 -> cleanUnquotedContent (from chunk_461)
// qd -> checkBashCommand
// $m5 -> parseCommand

// Telemetry/Logging placeholder
function logSecurityEvent(event: string, data: any) {
    // console.log(`[Security] ${event}`, data);
}

// --- Helper Functions (ported from chunk_461.ts to ensure self-contained logic) ---

function extractCommandInfo(command: string, isJq: boolean = false) {
    let withDoubleQuotes = "";
    let fullyUnquoted = "";
    let inEscape = false;
    let inSingleQuote = false;
    let inDoubleQuote = false;

    for (let i = 0; i < command.length; i++) {
        let char = command[i];

        if (inEscape) {
            inEscape = false;
            if (!inSingleQuote) withDoubleQuotes += char;
            if (!inSingleQuote && !inDoubleQuote) fullyUnquoted += char;
            continue;
        }

        if (char === "\\") {
            inEscape = true;
            if (!inSingleQuote) withDoubleQuotes += char;
            if (!inSingleQuote && !inDoubleQuote) fullyUnquoted += char;
            continue;
        }

        if (char === "'" && !inDoubleQuote) {
            inSingleQuote = !inSingleQuote;
            continue;
        }

        if (char === '"' && !inSingleQuote) {
            inDoubleQuote = !inDoubleQuote;
            if (!isJq) continue;
        }

        if (!inSingleQuote) withDoubleQuotes += char;
        if (!inSingleQuote && !inDoubleQuote) fullyUnquoted += char;
    }

    return {
        withDoubleQuotes,
        fullyUnquoted
    };
}

function cleanUnquotedContent(content: string) {
    return content
        .replace(/\s+2\s*>&\s*1(?=\s|$)/g, "")
        .replace(/[012]?\s*>\s*\/dev\/null/g, "")
        .replace(/\s*<\s*\/dev\/null/g, "");
}

// --- Security Checks ---

export function checkIfsInjection(command: string) {
    // ou5
    if (/\$IFS|\$\{[^}]*IFS/.test(command)) {
        logSecurityEvent("tengu_bash_security_check_triggered", {
            checkId: BASH_SECURITY_CHECKS.IFS_INJECTION,
            subId: 1
        });
        return { behavior: "ask", message: "Command contains IFS variable usage which could bypass security validation" };
    }
    return { behavior: "passthrough", message: "No IFS injection detected" };
}

export function checkProcEnvironAccess(command: string) {
    // ru5
    if (/\/proc\/.*\/environ/.test(command)) {
        logSecurityEvent("tengu_bash_security_check_triggered", {
            checkId: BASH_SECURITY_CHECKS.PROC_ENVIRON_ACCESS,
            subId: 1
        });
        return { behavior: "ask", message: "Command accesses /proc/*/environ which could expose sensitive environment variables" };
    }
    return { behavior: "passthrough", message: "No /proc/environ access detected" };
}

export function checkObfuscatedFlags(arg: { originalCommand: string, baseCommand?: string, fullyUnquotedContent: string }) {
    // su5
    const { originalCommand, baseCommand } = arg;
    const fullyUnquotedContent = arg.fullyUnquotedContent;
    const hasPipeOrChain = /[|&;]/.test(originalCommand);

    if (baseCommand === "echo" && !hasPipeOrChain) {
        return { behavior: "passthrough", message: "echo command is safe and has no dangerous flags" };
    }

    if (/\$'[^']*'/.test(originalCommand)) {
        logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 5 });
        return { behavior: "ask", message: "Command contains ANSI-C quoting which can hide characters" };
    }

    if (/\$"[^"]*"/.test(originalCommand)) {
        logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 6 });
        return { behavior: "ask", message: "Command contains locale quoting which can hide characters" };
    }

    if (/\$['"]{2}\s*-/.test(originalCommand)) {
        logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 9 });
        return { behavior: "ask", message: "Command contains empty special quotes before dash (potential bypass)" };
    }

    if (/(?:^|\s)(?:''|"")+\s*-/.test(originalCommand)) {
        logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 7 });
        return { behavior: "ask", message: "Command contains empty quotes before dash (potential bypass)" };
    }

    // Manual parser loop to detect quoted flags
    let inSingle = false;
    let inDouble = false;
    let escaped = false;

    for (let i = 0; i < originalCommand.length - 1; i++) {
        let char = originalCommand[i];
        let nextChar = originalCommand[i + 1];

        if (escaped) {
            escaped = false;
            continue;
        }
        if (char === "\\") {
            escaped = true;
            continue;
        }
        if (char === "'" && !inDouble) {
            inSingle = !inSingle;
            continue;
        }
        if (char === '"' && !inSingle) {
            inDouble = !inDouble;
            continue;
        }
        if (inSingle || inDouble) continue;

        // Check for quoted flag-like patterns
        if (char && nextChar && /\s/.test(char) && /['"`]/.test(nextChar)) {
            let quote = nextChar;
            let j = i + 2;
            let content = "";
            while (j < originalCommand.length && originalCommand[j] !== quote) {
                content += originalCommand[j];
                j++;
            }
            if (j < originalCommand.length && originalCommand[j] === quote && content.startsWith("-")) {
                logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 4 });
                return { behavior: "ask", message: "Command contains quoted characters in flag names" };
            }
        }

        // Check for quotes inside flag arguments
        if (char && nextChar && /\s/.test(char) && nextChar === "-") {
            let j = i + 1;
            let flagPart = "";
            while (j < originalCommand.length) {
                let c = originalCommand[j];
                if (!c) break;
                if (/[\s=]/.test(c)) break;
                if (/['"`]/.test(c)) {
                    // Exceptions for cut -d
                    if (baseCommand === "cut" && flagPart === "-d" && /['"`]/.test(c)) break;
                    // Lookahead
                    if (j + 1 < originalCommand.length) {
                        let n = originalCommand[j + 1];
                        if (n && !/[a-zA-Z0-9_'"-]/.test(n)) break;
                    }
                }
                flagPart += c;
                j++;
            }
            if (flagPart.includes('"') || flagPart.includes("'")) {
                logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 1 });
                return { behavior: "ask", message: "Command contains quoted characters in flag names" };
            }
        }
    }

    if (/\s['"`]-/.test(fullyUnquotedContent)) {
        logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 2 });
        return { behavior: "ask", message: "Command contains quoted characters in flag names" };
    }

    if (/['"`]{2}-/.test(fullyUnquotedContent)) {
        logSecurityEvent("tengu_bash_security_check_triggered", { checkId: BASH_SECURITY_CHECKS.OBFUSCATED_FLAGS, subId: 3 });
        return { behavior: "ask", message: "Command contains quoted characters in flag names" };
    }

    return { behavior: "passthrough", message: "No obfuscated flags detected" };
}

// --- Tree Sitter Integration ---

let parser: Parser | null = null;
let bashLanguage: Language | null = null;
let initializationPromise: Promise<void> | null = null;

export async function initializeTreeSitter() {
    if (initializationPromise) return initializationPromise;

    initializationPromise = (async () => {
        try {
            await Parser.init();
            parser = new Parser();

            // Allow loading WASM from different locations
            const wasmPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../../vendor/tree-sitter-bash.wasm');
            // Check if file exists, otherwise try other locations or embedded
            if (fs.existsSync(wasmPath)) {
                bashLanguage = await Language.load(wasmPath);
            } else {
                // Fallback or error - for now assume it might be in root vendor
                const rootVendorPath = path.resolve(process.cwd(), 'vendor/tree-sitter-bash.wasm');
                if (fs.existsSync(rootVendorPath)) {
                    bashLanguage = await Language.load(rootVendorPath);
                } else {
                    // Try 2.0.76 ?
                    console.warn("Tree-sitter WASM not found in expected paths");
                    throw new Error("Tree-sitter WASM not found");
                }
            }

            if (bashLanguage) {
                parser.setLanguage(bashLanguage);
            }
        } catch (e) {
            console.error("Failed to initialize TreeSitter:", e);
        }
    })();
    return initializationPromise;
}

export async function ensureInitialized() {
    if (!initializationPromise) {
        await initializeTreeSitter();
    }
    await initializationPromise;
}

export type ParseResult = {
    tree: Tree;
    rootNode: any; // WebTreeSitter uses SyntaxNode (Node)
    envVars: string[];
    commandNode: any | null;
    originalCommand: string;
};

function traverseForEnvVars(node: any): string[] {
    // Um5
    if (!node || node.type !== 'command') return [];
    const envs: string[] = [];
    for (const child of node.children) {
        if (!child) continue;
        if (child.type === 'variable_assignment') {
            envs.push(child.text);
        } else if (child.type === 'command_name' || child.type === 'word') {
            break;
        }
    }
    return envs;
}

function findCommandNode(node: any): any | null {
    // Ek2
    const children = node.children;

    // JE0 = Set(["command", "declaration_command"])
    const commandTypes = new Set(["command", "declaration_command"]);

    if (commandTypes.has(node.type)) return node;

    if (node.type === "variable_assignment" && node.parent) {
        // Siblings? Logic in chunk_462: return G.children.find(...)
        // In tree-sitter, we might need to check parent's children
        // But the logic in chunk_462 uses `G` which seems to be parent?
        // original: if (Q === "variable_assignment" && G) return G.children.find(...)
        // Here node.parent is G.
        return node.parent?.children.find((c: any) => c && commandTypes.has(c.type) && c.startIndex > node.startIndex) || null;
    }

    if (node.type === "pipeline" || node.type === "redirected_statement") {
        return children.find((c: any) => c && commandTypes.has(c.type)) || null;
    }

    for (const child of children) {
        const found = child && findCommandNode(child);
        if (found) return found;
    }

    return null;
}

export async function parseCommand(command: string): Promise<ParseResult | null> {
    // $m5
    await ensureInitialized();
    if (!command || command.length > 10000 || !parser || !bashLanguage) return null;

    try {
        const tree = parser.parse(command);
        if (!tree) return null;
        const rootNode = tree.rootNode;
        if (!rootNode) return null;

        const commandNode = findCommandNode(rootNode);
        const envVars = commandNode ? traverseForEnvVars(commandNode) : [];

        return {
            tree,
            rootNode,
            envVars,
            commandNode,
            originalCommand: command
        };
    } catch (e) {
        return null;
    }
}

// Export classes if needed for type compatibility, though web-tree-sitter uses its own
export { Parser };
export type { Tree, SyntaxNode as Node, TreeCursor };
