/**
 * File: src/utils/terminal/Suggestions.ts
 * Role: Utilities for generating command and slash-command suggestions.
 */

import { Fuse } from '../../vendor/Fuse.js';

export interface Suggestion {
    id: string;
    displayText: string;
    description: string;
    metadata?: any;
}

export interface Command {
    name: string;
    description: string;
    isHidden: boolean;
    userFacingName(): string;
    aliases?: string[];
    type?: string;
    argNames?: string[];
}

/**
 * Generates slash command suggestions based on user input.
 * 
 * @param {string} input - The user's input string.
 * @param {Command[]} allCommands - An array of command objects.
 * @returns {Suggestion[]} An array of suggestion objects.
 */
export function getSlashCommandSuggestions(input: string, allCommands: Command[]): Suggestion[] {
    if (!input.startsWith("/")) {
        return [];
    }

    const query = input.slice(1).toLowerCase().trim();

    if (query === "") {
        // Return all non-hidden commands sorted by userFacingName
        return allCommands
            .filter(c => !c.isHidden)
            .sort((a, b) => a.userFacingName().localeCompare(b.userFacingName()))
            .map(c => formatCommandSuggestion(c));
    }

    const commandData = allCommands
        .filter(c => !c.isHidden)
        .map(c => ({
            name: c.userFacingName(),
            description: c.description,
            command: c,
            aliases: c.aliases || []
        }));

    const fuse = new Fuse(commandData, {
        threshold: 0.3,
        keys: [
            { name: "name", weight: 3 },
            { name: "aliases", weight: 2 },
            { name: "description", weight: 0.5 }
        ]
    });

    const results = fuse.search(query);
    return results.map((r) => formatCommandSuggestion(r.item.command));
}

/**
 * Formats a command into a suggestion object.
 * 
 * @param {Command} command - The command object.
 * @param {string} [alias] - An optional alias for the command.
 * @returns {Suggestion} A suggestion object.
 */
export function formatCommandSuggestion(command: Command, alias?: string): Suggestion {
    const name = command.userFacingName();
    const aliasText = alias ? ` (${alias})` : "";
    return {
        id: command.name,
        displayText: `/${name}${aliasText}`,
        description: command.description + (command.type === "prompt" && command.argNames?.length ? ` (arguments: ${command.argNames.join(", ")})` : ""),
        metadata: command
    };
}
