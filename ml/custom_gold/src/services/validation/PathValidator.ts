
// Logic from chunk_463.ts (Path Validation, Shell Parsing)

import path from "path";
import os from "os";

// --- Parser Logic (bX1, Ck2) ---
// Stubbing complex parser logic with simplified versions
export const BashParser = {
    async parse(command: string) {
        return new ParsedCommand(command);
    }
};

class ParsedCommand {
    originalCommand: string;
    constructor(command: string) {
        this.originalCommand = command;
    }

    getPipeSegments() {
        return this.originalCommand.split("|").map(s => s.trim());
    }

    withoutOutputRedirections() {
        return this.originalCommand.replace(/>\s*\S+/g, "").trim();
    }

    getOutputRedirections() {
        const matches = this.originalCommand.matchAll(/>\s*(\S+)/g);
        const redirections = [];
        for (const match of matches) {
            redirections.push({ target: match[1], operator: ">" });
        }
        return redirections;
    }
}

// --- Path Validation Logic (Mk2, Tm5, vm5, bm5) ---
export function validatePathAccess(targetPath: string, cwd: string, mode: "read" | "write" | "create") {
    // Stub
    if (targetPath.includes("..")) {
        return { allowed: false, decisionReason: { type: "rule", rule: "Traversal" } };
    }
    return { allowed: true, resolvedPath: path.resolve(cwd, targetPath) };
}

export function validateCommandPaths(command: string, cwd: string, permissions: any, hasCd: boolean) {
    if (hasCd && (command.includes(">") || command.includes("write"))) {
        return {
            behavior: "ask",
            message: "Compound command with cd and write operation requires approval",
            decisionReason: { type: "other", reason: "Compound cd + write" }
        };
    }

    // Stub: Check dangerous commands
    if (command.startsWith("rm ")) {
        return {
            behavior: "ask",
            message: "Dangerous removal command",
            decisionReason: { type: "other", reason: "Dangerous command" }
        }
    }

    return { behavior: "passthrough", message: "Paths validated" };
}

export function validateRedirections(redirections: any[], cwd: string, permissions: any, hasCd: boolean) {
    if (hasCd && redirections.length > 0) {
        return {
            behavior: "ask",
            message: "Redirection with cd requires approval",
            decisionReason: { type: "other", reason: "Redirection with cd" }
        };
    }
    // Stub
    return { behavior: "passthrough", message: "Redirections validated" };
}

// --- Utils (Lk2, Rm5, Ok2, jm5) ---
export function expandHomeDir(p: string) {
    if (p === "~" || p.startsWith("~/")) {
        return path.join(os.homedir(), p.slice(1));
    }
    return p;
}

export function isHomeDir(p: string) {
    return p === os.homedir();
}

export function getParentDir(p: string) {
    return path.dirname(p);
}
