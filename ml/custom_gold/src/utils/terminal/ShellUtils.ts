
/**
 * Utilities for shell command manipulation and safety.
 * Logic from chunk_495.ts
 */

/**
 * Checks if a command uses a heredoc.
 * Deobfuscated from N$0 in chunk_495:522
 */
export function hasHeredoc(command: string): boolean {
    if (/\d\s*<<\s*\d/.test(command) || /\[\[\s*\d+\s*<<\s*\d+\s*\]\]/.test(command) || /\$\(\(.*<<.*\)\)/.test(command)) return false;
    return /<<-?\s*(?:(['"]?)(\w+)\1|\\(\w+))/.test(command);
}

/**
 * Checks if a command has multi-line quotes.
 * Deobfuscated from Sr5 in chunk_495:527
 */
export function hasMultilineQuotes(command: string): boolean {
    const singleQuoteMultiline = /'(?:[^'\\]|\\.)*\n(?:[^'\\]|\\.)*'/;
    const doubleQuoteMultiline = /"(?:[^"\\]|\\.)*\n(?:[^"\\]|\\.)*"/;
    return singleQuoteMultiline.test(command) || doubleQuoteMultiline.test(command);
}

/**
 * Wraps a command to prevent it from hanging on stdin.
 * Deobfuscated from kc2 in chunk_495:533
 */
export function wrapCommandWithNoStdin(command: string, redirect: boolean = true): string {
    if (hasHeredoc(command) || hasMultilineQuotes(command)) {
        // Use single quotes and escape internal single quotes
        const escaped = `'${command.replace(/'/g, "'\"'\"'")}'`;
        if (hasHeredoc(command)) return escaped;
        return redirect ? `${escaped} < /dev/null` : escaped;
    }

    // Fallback: simple shell escaping and redirection
    if (redirect) return `${command} < /dev/null`;
    return command;
}

/**
 * Checks if a command already has an input redirection.
 * Deobfuscated from xr5 in chunk_495:543
 */
export function hasInputRedirection(command: string): boolean {
    return /(?:^|[\s;&|])<(?![<(])\s*\S+/.test(command);
}

/**
 * Heuristic to check if a command is "safe" to append redirection to.
 * Deobfuscated from bc2 in chunk_495:547
 */
export function isSafeForRedirection(command: string): boolean {
    if (hasHeredoc(command)) return false;
    if (hasInputRedirection(command)) return false;
    return true;
}

/**
 * Intelligently injects < /dev/null into a command string.
 * Deobfuscated from gc2 in chunk_495:556
 */
export function injectNoStdin(command: string): string {
    if (command.includes("`")) return `${command} < /dev/null`;

    // Simplified logic: if it has pipes, we want to redirect the FIRST command
    // and let the pipes handle the rest. Or just redirect at the end?
    // In original code, it splits by tokens and finds the first pipe.

    if (command.includes("|")) {
        const parts = command.split("|");
        parts[0] = parts[0].trim() + " < /dev/null";
        return parts.join(" | ");
    }

    return `${command} < /dev/null`;
}
