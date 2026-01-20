import { createHash } from "node:crypto";
import { join } from "node:path";
import fs from "node:fs";
import { getOriginalCwd } from '../session/sessionStore.js';
import { logTelemetryEvent } from '../telemetry/telemetryInit.js';
import { getSettings } from '../terminal/settings.js';
import { parse, Token } from "../../vendor/shell-quote.js";
import { getMaxBashOutputLength } from "../../utils/terminal/BashConfig.js";

// Logic from chunk_536.ts and chunk_581.ts

export type BashSafetyBehavior = 'allow' | 'ask' | 'passthrough';

export interface BashSafetyResult {
    behavior: BashSafetyBehavior;
    message?: string;
    updatedInput?: any;
}

export const SEARCH_COMMANDS = new Set(["find", "grep", "rg", "ag", "ack", "locate", "which", "whereis"]);
export const READ_COMMANDS = new Set(["cat", "head", "tail", "less", "more", "wc", "stat", "file", "ls", "tree", "du"]);
export const BACKGROUND_EXCLUDED_COMMANDS = ["sleep"];

const V97 = 5000; // Summarization threshold

const COMMAND_DEFINITIONS: Record<string, {
    safeFlags: Record<string, 'none' | 'number' | 'string' | '{}' | 'char' | 'EOF'>;
    regex?: RegExp;
    additionalCommandIsDangerousCallback?: (cmd: string) => boolean;
}> = {
    'xargs': {
        safeFlags: {
            '-I': '{}',
            '-i': 'none',
            '-n': 'number',
            '-P': 'number',
            '-L': 'number',
            '-s': 'number',
            '-E': 'EOF',
            '-e': 'EOF',
            '-0': 'none',
            '-t': 'none',
            '-r': 'none',
            '-x': 'none',
            '-d': 'char'
        }
    },
    'git diff': {
        safeFlags: {
            '--stat': 'none',
            '--numstat': 'none',
            '--shortstat': 'none',
            '--name-only': 'none',
            '--name-status': 'none',
            '--color': 'none',
            '--no-color': 'none',
            '--dirstat': 'none',
            '--summary': 'none',
            '--patch-with-stat': 'none',
            '--word-diff': 'none',
            '--word-diff-regex': 'string',
            '--color-words': 'none',
            '--no-renames': 'none',
            '--no-ext-diff': 'none',
            '--check': 'none',
            '--ws-error-highlight': 'string',
            '--full-index': 'none',
            '--binary': 'none',
            '--abbrev': 'number',
            '--break-rewrites': 'none',
            '--find-renames': 'none',
            '--find-copies': 'none',
            '--find-copies-harder': 'none',
            '--irreversible-delete': 'none',
            '--diff-algorithm': 'string',
            '--histogram': 'none',
            '--patience': 'none',
            '--minimal': 'none',
            '--ignore-space-at-eol': 'none',
            '--ignore-space-change': 'none',
            '--ignore-all-space': 'none',
            '--ignore-blank-lines': 'none',
            '--inter-hunk-context': 'number',
            '--function-context': 'none',
            '--exit-code': 'none',
            '--quiet': 'none',
            '--cached': 'none',
            '--staged': 'none',
            '--pickaxe-regex': 'none',
            '--pickaxe-all': 'none',
            '--no-index': 'none',
            '--relative': 'string',
            '--diff-filter': 'string',
            '-p': 'none',
            '-u': 'none'
        }
    }
    // ... many more in original but these are the ones cited in 536
};

const SAFE_COMMAND_PREFIXES = [
    'date', 'cal', 'uptime', 'head', 'tail', 'wc', 'stat', 'strings',
    'hexdump', 'od', 'nl', 'id', 'uname', 'free', 'df', 'du', 'locale',
    'hostname', 'groups', 'nproc', 'docker ps', 'docker images', 'info',
    'basename', 'dirname', 'realpath', 'cut', 'tr', 'column', 'diff',
    'true', 'false', 'sleep', 'which', 'type'
];

const SAFE_COMMAND_REGEXES = [
    /^echo(?:\s+(?:'[^']*'|"[^"$<>\n\r]*"|[^|;&`$(){}><#\\!"'\s]+))*(?:\s+2>&1)?\s*$/,
    /^claude -h$/,
    /^claude --help$/,
    /^git status(?:\s|$)[^<>()$`|{}&;\n\r]*$/,
    /^git blame(?:\s|$)[^<>()$`|{}&;\n\r]*$/,
    /^git ls-files(?:\s|$)[^<>()$`|{}&;\n\r]*$/,
    /^git config --get[^<>()$`|{}&;\n\r]*$/,
    /^git remote -v$/,
    /^git remote show\s+[a-zA-Z0-9_-]+$/,
    /^git tag$/,
    /^git tag -l[^<>()$`|{}&;\n\r]*$/,
    /^git branch$/,
    /^git branch (?:-v|-vv|--verbose)$/,
    /^git branch (?:-a|--all)$/,
    /^git branch (?:-r|--remotes)$/,
    /^git branch (?:-l|--list)(?:\s+".*"|'[^']*')?$/,
    /^git branch (?:--color|--no-color|--column|--no-column)$/,
    /^git branch --sort=\S+$/,
    /^git branch --show-current$/,
    /^git branch (?:--contains|--no-contains)\s+\S+$/,
    /^git branch (?:--merged|--no-merged)(?:\s+\S+)?$/,
    /^uniq(?:\s+(?:-[a-zA-Z]+|--[a-zA-Z-]+(?:=\S+)?|-[fsw]\s+\d+))*(?:\s|$)\s*$/,
    /^pwd$/,
    /^whoami$/,
    /^node -v$/,
    /^npm -v$/,
    /^python --version$/,
    /^python3 --version$/,
    /^tree$/,
    /^history(?:\s+\d+)?\s*$/,
    /^alias$/,
    /^arch(?:\s+(?:--help|-h))?\s*$/,
    /^ip addr$/,
    /^ifconfig(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?\s*$/,
    /^jq(?!\s+.*(?:-f\b|--from-file|--rawfile|--slurpfile|--run-tests|-L\b|--library-path))(?:\s+(?:-[a-zA-Z]+|--[a-zA-Z-]+(?:=\S+)?))*(?:\s+'[^'`]*'|\s+"[^"`]*"|\s+[^-\s'"][^\s]*)+\s*$/,
    /^cd(?:\s+(?:'[^']*'|"[^"]*"|[^\s;|&`$(){}><#\\]+))?$/,
    /^ls(?:\s+[^<>()$`|{}&;\n\r]*)?$/,
    /^find(?:\s+(?:\\[()]|(?!-delete\b|-exec\b|-execdir\b|-ok\b|-okdir\b|-fprint0?\b|-fls\b|-fprintf\b)[^<>()$`|{}&;\n\r\s]|\s)+)?$/
];

const DANGEROUS_PATTERNS = [
    { pattern: /<\(/, message: 'process substitution <()' },
    { pattern: />\(/, message: 'process substitution >()' },
    { pattern: /\$\(/, message: '$() command substitution' },
    { pattern: /\$\{/, message: '${} parameter substitution' },
    { pattern: /~\[/, message: 'Zsh-style parameter expansion' },
    { pattern: /\(e:/, message: 'Zsh-style glob qualifiers' },
    { pattern: /<#/, message: 'PowerShell comment syntax' }
];

export function evaluateBashCommandSafety(command: string): BashSafetyResult {
    const trimmedCmd = command.trim();

    if (!trimmedCmd) {
        return { behavior: 'allow', message: 'Empty command is safe' };
    }

    const tokensResult = parse(command, (key) => `$${key}`);
    if (!tokensResult) {
        return { behavior: 'passthrough', message: 'Command cannot be parsed, requires further permission checks' };
    }

    if (isCommandSafe(command)) {
        return { behavior: 'allow' };
    }

    if (/\$IFS|\$\{[^}]*IFS/.test(command)) {
        return { behavior: 'ask', message: 'Command contains IFS variable usage which could bypass security validation' };
    }

    if (/\/proc\/.*\/environ/.test(command)) {
        return { behavior: 'ask', message: 'Command accesses /proc/*/environ which could expose sensitive environment variables' };
    }

    if (checkObfuscatedFlags(command)) {
        return { behavior: 'ask', message: 'Command contains potential obfuscation in flag names' };
    }

    if (/^\s*\t/.test(command)) return { behavior: 'ask', message: 'Command appears to be an incomplete fragment (starts with tab)' };
    if (trimmedCmd.startsWith('-')) return { behavior: 'ask', message: 'Command appears to be an incomplete fragment (starts with flags)' };
    if (/^\s*(&&|\|\||;|>>?|<)/.test(command)) return { behavior: 'ask', message: 'Command appears to be a continuation line (starts with operator)' };

    const { withDoubleQuotes } = unquoteSpecial(command);

    if (checkUnescapedChar(withDoubleQuotes, '`')) {
        return { behavior: 'ask', message: 'Command contains backticks (`) for command substitution' };
    }

    for (const { pattern, message } of DANGEROUS_PATTERNS) {
        if (pattern.test(withDoubleQuotes)) {
            return { behavior: 'ask', message: `Command contains ${message}` };
        }
    }

    const { fullyUnquoted } = unquoteSpecial(command);
    if (/</.test(fullyUnquoted)) {
        return { behavior: 'ask', message: 'Command contains input redirection (<) which could read sensitive files' };
    }

    return { behavior: 'passthrough', message: 'Command passed security validation but requires manual permission' };
}

function isCommandSafe(cmd: string): boolean {
    const trimmed = cmd.trim();
    const normalized = trimmed.endsWith(' 2>&1') ? trimmed.slice(0, -5).trim() : trimmed;

    if (process.platform === 'win32') {
        if (/\\\\[a-zA-Z0-9._\-:[\]%]+(?:@(?:\d+|ssl))?\\/i.test(normalized)) return false;
        if (/\/\/[a-zA-Z0-9._\-:[\]%]+(?:@(?:\d+|ssl))?\//i.test(normalized)) return false;
    }

    for (const regex of SAFE_COMMAND_REGEXES) {
        if (regex.test(normalized)) return true;
    }

    for (const prefix of SAFE_COMMAND_PREFIXES) {
        const regex = new RegExp(`^${prefix}(?:\\s|$)[^<>()$\`|{}&;\\n\\r]*$`);
        if (regex.test(normalized)) return true;
    }

    return false;
}

function checkObfuscatedFlags(cmd: string): boolean {
    if (/\$'[^']*'/.test(cmd)) return true;
    if (/\$"[^"]*"/.test(cmd)) return true;
    if (/\$['"]{2}\s*-/.test(cmd)) return true;
    if (/(?:^|\s)(?:''|"")+\s*-/.test(cmd)) return true;
    return false;
}

function unquoteSpecial(cmd: string) {
    let withDoubleQuotes = '';
    let fullyUnquoted = '';
    let inSingleQuote = false;
    let inDoubleQuote = false;
    let escaped = false;

    for (let i = 0; i < cmd.length; i++) {
        const char = cmd[i];
        if (escaped) {
            escaped = false;
            if (!inSingleQuote) withDoubleQuotes += char;
            if (!inSingleQuote && !inDoubleQuote) fullyUnquoted += char;
            continue;
        }
        if (char === '\\') {
            escaped = true;
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
            continue;
        }
        if (!inSingleQuote) withDoubleQuotes += char;
        if (!inSingleQuote && !inDoubleQuote) fullyUnquoted += char;
    }

    return { withDoubleQuotes, fullyUnquoted };
}

function checkUnescapedChar(text: string, char: string): boolean {
    let escaped = false;
    for (let i = 0; i < text.length; i++) {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (text[i] === '\\') {
            escaped = true;
            continue;
        }
        if (text[i] === char) return true;
    }
    return false;
}

export async function summarizeBashOutput(
    stdout: string,
    command: string,
    context: string = "",
    signal?: AbortSignal
): Promise<{ shouldSummarize: boolean, summary?: string, rawOutputPath?: string, reason?: string }> {
    const combinedOutput = stdout; // Original deobfuscation handled stdout/stderr combination elsewhere
    const { isImage } = truncateOutput(combinedOutput);
    if (isImage) return { shouldSummarize: false, reason: "image_data" };

    if (combinedOutput.length < V97) {
        return { shouldSummarize: false, reason: "below_threshold" };
    }

    // In a real implementation this would call an LLM (jK in chunk_536)
    // Here we stub the logic since we don't have the LLM caller directly available in this service
    // But the deobfuscated chunk shows it saves the raw output anyway.

    const rawOutputPath = saveRawOutput(stdout, "", command);
    return {
        shouldSummarize: false, // Default to false if we can't call LLM
        reason: "llm_summarization_not_implemented_in_stub",
        rawOutputPath
    };
}

function saveRawOutput(stdout: string, stderr: string, command: string): string {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const hash = createHash("sha256").update(command).digest("hex").slice(0, 8);
    const filename = `${timestamp}-${hash}.txt`;
    const dir = join(getOriginalCwd(), "bash-outputs");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const filePath = join(dir, filename);
    fs.writeFileSync(filePath, `COMMAND: ${command}\n\nSTDOUT:\n${stdout}\n\nSTDERR:\n${stderr}`);
    return filePath;
}

export function trimOutput(output: string): string {
    return output.trim();
}

export function truncateOutput(output: string, maxLength: number = 5000) {
    const isImage = /^data:image\/[a-z0-9.+_-]+;base64,/i.test(output);
    const limit = maxLength || getMaxBashOutputLength();

    if (isImage) {
        return { content: output, truncated: false, isImage: true, totalLines: 1 };
    }

    if (output.length <= limit) {
        return { content: output, truncated: false, isImage: false, totalLines: output.split('\n').length };
    }

    const truncated = output.slice(0, limit);
    const remainingLines = output.slice(limit).split('\n').length;
    return {
        content: `${truncated}\n\n... [${remainingLines} lines truncated] ...`,
        truncated: true,
        isImage: false,
        totalLines: output.split('\n').length
    };
}
