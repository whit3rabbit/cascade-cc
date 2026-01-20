/**
 * Utility to parse and wrap commands with prefixes.
 * Deobfuscated from YFB, xg, and related in chunk_216.ts.
 */

export type CommandType = "bash" | "background" | "prompt";

/**
 * Wraps a command with its prefix.
 */
export function wrapCommandWithPrefix(command: string, type: CommandType): string {
    switch (type) {
        case "bash":
            return `!${command}`;
        case "background":
            return `&${command}`;
        default:
            return command;
    }
}

/**
 * Detects the command type from its prefix.
 */
export function getCommandType(command: string): CommandType {
    if (command.startsWith("!")) return "bash";
    if (command.startsWith("&")) return "background";
    return "prompt";
}

/**
 * Strips the prefix from a command.
 */
export function stripCommandPrefix(command: string): string {
    const type = getCommandType(command);
    if (type === "prompt") return command;
    return command.slice(1);
}

/**
 * Checks if a character is a valid command prefix.
 */
export function isCommandPrefix(char: string): boolean {
    return char === "!" || char === "&";
}
