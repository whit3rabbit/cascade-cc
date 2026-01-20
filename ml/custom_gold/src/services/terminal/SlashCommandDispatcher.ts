
import { existsSync } from 'fs';

export interface SlashCommand {
    commandName: string;
    args: string;
    isMcp: boolean;
}

// Ld2
export function parseSlashCommand(input: string): SlashCommand | null {
    const trimmed = input.trim();
    if (!trimmed.startsWith('/')) return null;

    const parts = trimmed.slice(1).split(' ');
    if (!parts[0]) return null;

    let commandName = parts[0];
    let isMcp = false;
    let argStartIndex = 1;

    if (parts.length > 1 && parts[1] === '(MCP)') {
        commandName += ' (MCP)';
        isMcp = true;
        argStartIndex = 2;
    }

    const args = parts.slice(argStartIndex).join(' ');

    return {
        commandName,
        args,
        isMcp
    };
}

function isValidCommandIdentifier(name: string): boolean {
    return !/[^a-zA-Z0-9:\-_]/.test(name);
}

// Od2 - Simplified dispatch logic
export async function dispatchSlashCommand(
    input: string,
    precedingInputBlocks: any[],
    history: any[],
    allowedCommands: Set<string>,
    // other dependencies...
) {
    const parsed = parseSlashCommand(input);
    if (!parsed) {
        return {
            messages: [
                // Error message about slash command format
                { type: 'user', content: "Commands are in the form `/command [args]`" }
            ],
            shouldQuery: false
        };
    }

    const { commandName, args, isMcp } = parsed;

    // Check if command is known
    // Mocking the command registry check for now
    const isKnown = allowedCommands.has(commandName);

    if (!isKnown) {
        // Special case: check if it's a file path (for executing scripts/files as commands)
        // But only if it's a valid identifier and not an existing file at root
        const existsAtRoot = existsSync(`/${commandName}`);
        if (isValidCommandIdentifier(commandName) && !existsAtRoot) {
            return {
                messages: [{ type: 'user', content: `Unknown slash command: ${commandName}` }],
                shouldQuery: false
            };
        }

        // If not a slash command, treat as regular prompt
        return {
            messages: [{ type: 'user', content: input }],
            shouldQuery: true
        };
    }

    // Command execution logic would go here
    // ...

    return {
        messages: [],
        shouldQuery: false
    };
}
