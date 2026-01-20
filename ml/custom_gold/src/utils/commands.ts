/**
 * Checks if a command object matches a given name or alias.
 * Deobfuscated from xB3 in chunk_217.ts.
 */
export function matchesCommandName(command: { name: string; aliases?: string[] }, nameOrAlias: string): boolean {
    return command.name === nameOrAlias || (command.aliases?.includes(nameOrAlias) ?? false);
}

/**
 * Finds a command in a list by name or alias.
 * Deobfuscated from HA1 in chunk_217.ts.
 */
export function findCommandByName<T extends { name: string; aliases?: string[] }>(commands: T[], nameOrAlias: string): T | undefined {
    return commands.find((cmd) => matchesCommandName(cmd, nameOrAlias));
}

/**
 * Filters out hook progress messages from a list of messages.
 * Deobfuscated from pi in chunk_217.ts.
 */
export function filterHookProgress(messages: any[]): any[] {
    return messages.filter((msg) => msg.data?.type !== "hook_progress");
}
