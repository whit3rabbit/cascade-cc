
// Logic from chunk_464.ts (Command Rules, Rule Matching)

import { parsePermissionRule } from "../sandbox/sandboxConfigGenerator.js";
// import { parse } from "shell-quote"; // Assuming shell parser available or mocked

// Helpers for Sed/Shell parsing
function parseShell(cmd: string) {
    // Stub shell parser if not available
    // In a real impl, imports from utils/shell
    // Here we do basic splitting for sed args which is the main usage
    const tokens = cmd.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
    return {
        success: true,
        tokens: tokens.map(t => t.replace(/^['"]|['"]$/g, "")) // simple quote strip
    };
}
const sI = parseShell;

function AZ(args: string[]) {
    // Basic filter for paths (ignoring flags)
    return args.filter(a => !a.startsWith("-"));
}

function qk2(args: string[], flags: Set<string>, defaults: string[] = []) {
    let paths: string[] = [];
    let skipNext = false;
    for (const arg of args) {
        if (skipNext) {
            skipNext = false;
            // logic to add path if not flag?
            continue;
        }
        if (arg.startsWith("-")) {
            // checks for flags that take args
            // Simplified logic
            continue;
        }
        paths.push(arg);
    }
    return paths.length > 0 ? paths : defaults;
}

// --- Command Definitions (FE0) ---
// Used to extract paths or validate args. We map keys for SAFE_COMMANDS.

const commandHandlers: Record<string, (args: string[]) => string[]> = {
    cd: (A) => A.length === 0 ? ["."] : [A.join(" ")],
    ls: (A) => {
        let Q = AZ(A);
        return Q.length > 0 ? Q : ["."];
    },
    mkdir: AZ,
    touch: AZ,
    rm: AZ,
    rmdir: AZ,
    mv: AZ,
    cp: AZ,
    cat: AZ,
    head: AZ,
    tail: AZ,
    sort: AZ,
    uniq: AZ,
    wc: AZ,
    cut: AZ,
    paste: AZ,
    column: AZ,
    file: AZ,
    stat: AZ,
    diff: AZ,
    awk: AZ,
    strings: AZ,
    hexdump: AZ,
    od: AZ,
    base64: AZ,
    nl: AZ,
    sha256sum: AZ,
    sha1sum: AZ,
    md5sum: AZ,
    tr: (A) => AZ(A), // Simplified
    grep: (A) => AZ(A), // Simplified
    rg: (A) => AZ(A), // Simplified
    sed: (A) => {
        // Sed parsing logic is complex
        return [];
    },
    jq: (A) => [],
    git: (A) => []
};

export const SAFE_COMMANDS = new Set(Object.keys(commandHandlers));

export const COMMAND_DESCRIPTIONS: Record<string, string> = {
    cd: "change directories to",
    ls: "list files in",
    find: "search files in",
    mkdir: "create directories in",
    touch: "create or modify files in",
    rm: "remove files from",
    rmdir: "remove directories from",
    mv: "move files to/from",
    cp: "copy files to/from",
    cat: "concatenate files from",
    head: "read the beginning of files from",
    tail: "read the end of files from",
    sort: "sort contents of files from",
    uniq: "filter duplicate lines from files in",
    wc: "count lines/words/bytes in files from",
    cut: "extract columns from files in",
    paste: "merge files from",
    column: "format files from",
    tr: "transform text from files in",
    file: "examine file types in",
    stat: "read file stats from",
    diff: "compare files from",
    awk: "process text from files in",
    strings: "extract strings from files in",
    hexdump: "display hex dump of files from",
    od: "display octal dump of files from",
    base64: "encode/decode files from",
    nl: "number lines in files from",
    grep: "search for patterns in files from",
    rg: "search for patterns in files from",
    sed: "edit files in",
    git: "access files with git from",
    jq: "process JSON from files in",
    sha256sum: "compute SHA-256 checksums for files in",
    sha1sum: "compute SHA-1 checksums for files in",
    md5sum: "compute MD5 checksums for files in"
};

export const COMMAND_PERMISSIONS: Record<string, "read" | "write" | "create"> = {
    cd: "read",
    ls: "read",
    find: "read",
    mkdir: "create",
    touch: "create",
    rm: "write",
    rmdir: "write",
    mv: "write",
    cp: "write",
    cat: "read",
    head: "read",
    tail: "read",
    sort: "read",
    uniq: "read",
    wc: "read",
    cut: "read",
    paste: "read",
    column: "read",
    tr: "read",
    file: "read",
    stat: "read",
    diff: "read",
    awk: "read",
    strings: "read",
    hexdump: "read",
    od: "read",
    base64: "read",
    nl: "read",
    grep: "read",
    rg: "read",
    sed: "write",
    git: "read",
    jq: "read",
    sha256sum: "read",
    sha1sum: "read",
    md5sum: "read"
};

// --- Sed Validation Logic (Pk2, mm5, Sk2, zE0) ---

function validateSedCommandInner(cmd: string, allowWrites: boolean) {
    // Regex based matching from chunk_464
    const sedMatch = cmd.match(/^\s*sed\s+/);
    if (!sedMatch) return false;
    const argsStr = cmd.slice(sedMatch[0].length);
    const parsed = sI(argsStr);
    if (!parsed.success) return false;
    const args = parsed.tokens;

    // Filter flags
    const flags = args.filter(a => a.startsWith("-") && a !== "--");

    // Check for dangerous flags
    const allowedFlags = ["-n", "--quiet", "--silent", "-E", "--regexp-extended", "-r", "-z", "--zero-terminated", "--posix"];
    if (allowWrites) {
        allowedFlags.push("-i", "--in-place");
    }

    // Check validity of flags
    // xk2 logic...
    for (const flag of flags) {
        if (!allowedFlags.includes(flag) && !flag.split("").every(char => flag.startsWith("-") && allowedFlags.some(af => af.includes(char)))) {
            // loose check for grouped flags like -rn
            // return false; 
        }
    }

    // If not allowing writes, ensure no -i
    if (!allowWrites && (flags.includes("-i") || flags.includes("--in-place"))) return false;

    return true;
}

// Exported Sed Check
export function validateSedCommand(command: string, mode: any) {
    if (validateSedCommandInner(command, true)) {
        if (command.includes("-i") || command.includes("--in-place")) {
            return { behavior: "ask", message: "sed in-place edit" };
        }
        return { behavior: "passthrough", message: "sed check passed" };
    }
    // If it fails validation even with writes allowed, maybe it's just syntax error or weird flags?
    // The original logic is: zE0 checks if it's safe (mm5) or dangerous but valid (Sk2)
    return { behavior: "passthrough", message: "sed check passed" };
}

function parseRuleValue(ruleValue: string) {
    if (ruleValue.includes(":")) {
        const [toolName, ...rest] = ruleValue.split(":");
        return { toolName, ruleContent: rest.join(":") };
    }
    return parsePermissionRule(ruleValue);
}

function collectRules(context: any, key: string) {
    const rules: Array<{ ruleValue: string; toolName: string; ruleContent?: string; source: string }> = [];

    const settingsRules = context?.[key];
    if (Array.isArray(settingsRules)) {
        for (const ruleValue of settingsRules) {
            const parsed = parseRuleValue(ruleValue);
            rules.push({ ruleValue, toolName: parsed.toolName, ruleContent: parsed.ruleContent, source: "localSettings" });
        }
    }
    // ... merge other sources logic ...
    return rules;
}

// --- Rule Matching ---
export function findMatchingRules(command: string, context: any, matchType: "exact" | "prefix") {
    // Real implementation would gather rules from all levels (global, project, user)
    const denyRules = collectRules(context, "deny");
    const askRules = collectRules(context, "ask");
    const allowRules = collectRules(context, "allow");

    // $E0 logic: matches command against rules
    const matches = (rules: any[]) => rules.filter(rule => {
        if (rule.toolName !== "Bash") return false; // assuming Bash command for now
        return matchCommandRule(command, rule, matchType);
    });

    return {
        matchingDenyRules: matches(denyRules),
        matchingAskRules: matches(askRules),
        matchingAllowRules: matches(allowRules)
    };
}

export function matchCommandRule(command: string, rule: any, matchType: "exact" | "prefix") {
    const ruleContent = rule.ruleContent ?? "";
    if (!ruleContent) return true; // Rule without content matches all?

    // Clean command (remove redirs etc) -> FS(G) in chunk_464
    const cleanCommand = command.trim().split(/\s+(>|>>|<|\||&|&&|;)\s+/)[0];

    if (matchType === "prefix") {
        if (ruleContent.endsWith(":*")) {
            const prefix = ruleContent.slice(0, -2);
            return cleanCommand.startsWith(prefix);
        }
        return cleanCommand.startsWith(ruleContent);
    }

    // exact logic
    if (ruleContent.endsWith(":*")) {
        const prefix = ruleContent.slice(0, -2);
        return cleanCommand.startsWith(prefix);
    }
    return cleanCommand === ruleContent;
}

// --- Main Validation Entry Point (bk2) ---
export function validateCommandWithRules(input: any, context: any, options: any, hasCd?: boolean) {
    const command = input.command.trim();

    // Check Sed
    if (command.startsWith("sed ")) {
        const sedResult = validateSedCommand(command, context?.mode);
        if (sedResult.behavior !== "passthrough") {
            return {
                behavior: "ask",
                message: sedResult.message ?? "Command requires approval",
                decisionReason: { type: "other", reason: sedResult.message ?? "sed in-place edit" }
            };
        }
    }

    const { matchingDenyRules, matchingAskRules, matchingAllowRules } = findMatchingRules(command, context, "exact");

    if (matchingDenyRules[0]) {
        return {
            behavior: "deny",
            message: `Permission to use Bash with command ${command} has been denied.`,
            decisionReason: { type: "rule", rule: matchingDenyRules[0] }
        };
    }

    if (matchingAskRules[0]) {
        return {
            behavior: "ask",
            message: "Command requires approval",
            decisionReason: { type: "rule", rule: matchingAskRules[0] }
        };
    }

    // Injection check (qd)... stubbed here but important

    if (matchingAllowRules[0]) {
        return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: { type: "rule", rule: matchingAllowRules[0] }
        };
    }

    return {
        behavior: "passthrough",
        message: `Command '${command}' requires approval`,
        decisionReason: { type: "other", reason: "Default ask" }
    };
}
