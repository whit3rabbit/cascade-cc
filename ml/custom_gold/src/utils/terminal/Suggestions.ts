import { Fuse, SearchResult } from '../../vendor/Fuse.js';

// Based on chunk_495.ts:1-78

export interface Suggestion {
    id: string;
    displayText: string;
    description: string;
    metadata?: any;
}

export function formatCommandSuggestion(command: any, alias?: string): Suggestion {
    const name = command.userFacingName();
    const aliasText = alias ? ` (${alias})` : "";
    return {
        id: command.name,
        displayText: `/${name}${aliasText}`,
        description: command.description + (command.type === "prompt" && command.argNames?.length ? ` (arguments: ${command.argNames.join(", ")})` : ""),
        metadata: command
    };
}

export function getSlashCommandSuggestions(input: string, allCommands: any[]): Suggestion[] {
    if (!input.startsWith("/")) return [];

    const query = input.slice(1).toLowerCase().trim();
    if (query === "") {
        // Return all non-hidden commands sorted by source/name
        return allCommands
            .filter(c => !c.isHidden)
            .sort((a, b) => a.userFacingName().localeCompare(b.userFacingName()))
            .map(c => formatCommandSuggestion(c));
    }

    const commandData = allCommands.filter(c => !c.isHidden).map(c => ({
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
    return results.map((r: SearchResult<any>) => formatCommandSuggestion(r.item.command));
}
