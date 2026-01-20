
/**
 * Defines which bash commands and flags are considered "safe" to run without explicit user confirmation.
 * Deobfuscated from H89 in chunk_840.ts.
 */

const GIT_COMMON_BRANCH_FLAGS = {
    "--all": "none",
    "--branches": "none",
    "--tags": "none",
    "--remotes": "none"
};

const GIT_COMMON_DATE_FLAGS = {
    "--since": "string",
    "--after": "string",
    "--until": "string",
    "--before": "string"
};

const GIT_LOG_DECORATE_FLAGS = {
    "--oneline": "none",
    "--graph": "none",
    "--decorate": "none",
    "--no-decorate": "none",
    "--date": "string",
    "--relative-date": "none"
};

const GIT_COMMON_COUNT_FLAGS = {
    "--max-count": "number",
    "-n": "number"
};

const GIT_DIFF_STAT_FLAGS = {
    "--stat": "none",
    "--numstat": "none",
    "--shortstat": "none",
    "--name-only": "none",
    "--name-status": "none"
};

const GIT_COLOR_FLAGS = {
    "--color": "none",
    "--no-color": "none"
};

const GIT_PATCH_FLAGS = {
    "--patch": "none",
    "-p": "none",
    "--no-patch": "none",
    "--no-ext-diff": "none",
    "-s": "none"
};

const GIT_LOG_FILTER_FLAGS = {
    "--author": "string",
    "--committer": "string",
    "--grep": "string"
};

export const BASH_SECURITY_POLICY: Record<string, any> = {
    "xargs": {
        safeFlags: {
            "-I": "{}", "-i": "none", "-n": "number", "-P": "number", "-L": "number",
            "-s": "number", "-E": "EOF", "-e": "EOF", "-0": "none", "-t": "none",
            "-r": "none", "-x": "none", "-d": "char"
        }
    },
    "git diff": {
        safeFlags: {
            ...GIT_DIFF_STAT_FLAGS,
            ...GIT_COLOR_FLAGS,
            "--dirstat": "none", "--summary": "none", "--patch-with-stat": "none",
            "--word-diff": "none", "--word-diff-regex": "string", "--color-words": "none",
            "--no-renames": "none", "--no-ext-diff": "none", "--check": "none",
            "--ws-error-highlight": "string", "--full-index": "none", "--binary": "none",
            "--abbrev": "number", "--break-rewrites": "none", "--find-renames": "none",
            "--find-copies": "none", "--find-copies-harder": "none", "--irreversible-delete": "none",
            "--diff-algorithm": "string", "--histogram": "none", "--patience": "none",
            "--minimal": "none", "--ignore-space-at-eol": "none", "--ignore-space-change": "none",
            "--ignore-all-space": "none", "--ignore-blank-lines": "none", "--inter-hunk-context": "number",
            "--function-context": "none", "--exit-code": "none", "--quiet": "none",
            "--cached": "none", "--staged": "none", "--pickaxe-regex": "none",
            "--pickaxe-all": "none", "--no-index": "none", "--relative": "string",
            "--diff-filter": "string", "-p": "none", "-u": "none", "-s": "none",
            "-M": "none", "-C": "none", "-B": "none", "-D": "none", "-l": "none",
            "-S": "none", "-G": "none", "-O": "none", "-R": "none"
        }
    },
    "git log": {
        safeFlags: {
            ...GIT_LOG_DECORATE_FLAGS,
            ...GIT_COMMON_BRANCH_FLAGS,
            ...GIT_COMMON_DATE_FLAGS,
            ...GIT_COMMON_COUNT_FLAGS,
            ...GIT_DIFF_STAT_FLAGS,
            ...GIT_COLOR_FLAGS,
            ...GIT_PATCH_FLAGS,
            ...GIT_LOG_FILTER_FLAGS,
            "--abbrev-commit": "none", "--full-history": "none", "--dense": "none",
            "--sparse": "none", "--simplify-merges": "none", "--ancestry-path": "none",
            "--source": "none", "--first-parent": "none", "--merges": "none",
            "--no-merges": "none", "--reverse": "none", "--walk-reflogs": "none",
            "--skip": "number", "--max-age": "number", "--min-age": "number",
            "--no-min-parents": "none", "--no-max-parents": "none", "--follow": "none",
            "--pretty": "string", "--format": "string", "--diff-filter": "string",
            "-S": "string", "-G": "string", "--pickaxe-regex": "none", "--pickaxe-all": "none"
        }
    },
    "git show": {
        safeFlags: {
            ...GIT_LOG_DECORATE_FLAGS, ...GIT_DIFF_STAT_FLAGS, ...GIT_COLOR_FLAGS, ...GIT_PATCH_FLAGS,
            "--abbrev-commit": "none", "--word-diff": "none", "--word-diff-regex": "string",
            "--color-words": "none", "--pretty": "string", "--first-parent": "none",
            "--diff-filter": "string", "-m": "none", "--quiet": "none"
        }
    },
    // ... Additional policies for sed, pip list, sort, man, help, npm list, netstat, ps, etc.
};

/**
 * A set of regexes for entire command lines that are always safe.
 * Deobfuscated from L97 in chunk_841.ts.
 */
export const ALWAYS_SAFE_COMMAND_REGEXES: RegExp[] = [
    /^echo(?:\s+(?:'[^']*'|"[^"$<>\n\r]*"|[^|;&`$(){}><#\\!"'\s]+))*(?:\s+2>&1)?\s*$/,
    /^claude -h$/,
    /^claude --help$/,
    /^git status(?:\s|$)[^<>()$`|{}&;\n\r]*$/,
    /^git blame(?:\s|$)[^<>()$`|{}&;\n\r]*$/,
    /^git ls-files(?:\s|$)[^<>()$`|{}&;\n\r]*$/,
    /^git config --get[^<>()$`|{}&;\n\r]*$/,
    /^git remote -v$/,
    // ... many others
];

export function isCommandSandboxedSafe(command: string): boolean {
    if (ALWAYS_SAFE_COMMAND_REGEXES.some(re => re.test(command))) return true;

    // Logic to check against BASH_SECURITY_POLICY
    const trimmed = command.trim();
    for (const [name, policy] of Object.entries(BASH_SECURITY_POLICY)) {
        if (trimmed.startsWith(name)) {
            // Basic flag validation logic...
        }
    }
    return false;
}
