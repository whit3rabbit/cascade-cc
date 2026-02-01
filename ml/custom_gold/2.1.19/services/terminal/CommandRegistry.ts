/**
 * File: src/services/terminal/CommandRegistry.ts
 * Role: Central registry for all CLI commands and slash commands.
 */

import { CommandDefinition } from '../../commands/helpers.js';
import { helpCommandDefinition, mobileCommandDefinition } from '../../commands/help.js';
import { initCommandDefinition } from '../../commands/init.js';

export class CommandRegistry {
    private commands: Map<string, CommandDefinition>;
    private aliases: Map<string, string>;

    constructor() {
        this.commands = new Map();
        this.aliases = new Map();
    }

    /**
     * Registers a command object.
     */
    register(command: CommandDefinition): void {
        this.commands.set(command.name, command);
        if (command.aliases) {
            command.aliases.forEach((alias: string) => {
                this.aliases.set(alias, command.name);
            });
        }
    }

    /**
     * Finds a command by name or alias.
     */
    findCommand(name: string): CommandDefinition | null | undefined {
        const cmdName = name.startsWith('/') ? name.slice(1) : name;
        if (this.commands.has(cmdName)) return this.commands.get(cmdName);
        const realName = this.aliases.get(cmdName);
        if (realName) return this.commands.get(realName);
        return null;
    }

    /**
     * Returns all registered commands.
     */
    getAllCommands(): CommandDefinition[] {
        return Array.from(this.commands.values());
    }

    /**
     * Returns commands filtered by eligibility and search term.
     */
    async getFilteredCommands(filter: string = ""): Promise<CommandDefinition[]> {
        const all = this.getAllCommands();
        return all.filter(cmd =>
            cmd.name.includes(filter) ||
            (cmd.description && cmd.description.includes(filter))
        );
    }
}

export const commandRegistry = new CommandRegistry();

/**
 * Initializes and registers all built-in commands.
 */
export async function initializeCommandDefinitions(): Promise<void> {
    console.log("Initializing commands...");

    // Register implemented commands
    commandRegistry.register(helpCommandDefinition as CommandDefinition);
    commandRegistry.register(mobileCommandDefinition as CommandDefinition);
    commandRegistry.register(initCommandDefinition as CommandDefinition);

    // Register simple stubs for other common commands to avoid 'Unknown command' errors
    // These will eventually be moved to their own files
    const stubs: Partial<CommandDefinition>[] = [
        {
            name: "clear",
            description: "Clear the conversation history",
            type: "local",
            userFacingName: () => "clear",
            isEnabled: () => true,
            isHidden: false,
            call: () => { /* Handled by REPL/SlashCommandDispatcher */ }
        },
        {
            name: "compact",
            description: "Compact conversation history to save tokens",
            type: "prompt",
            userFacingName: () => "compact",
            isEnabled: () => true,
            isHidden: false,
            getPromptForCommand: async () => [{ type: "text", text: "Please compact the conversation history." }]
        },
        {
            name: "cost",
            description: "Show current session cost and token usage",
            type: "local",
            userFacingName: () => "cost",
            isEnabled: () => true,
            isHidden: false,
            call: () => { /* Handled by REPL/SlashCommandDispatcher */ }
        },
        {
            name: "doctor",
            description: "Check the health of your Claude Code installation",
            type: "local",
            userFacingName: () => "doctor",
            isEnabled: () => true,
            isHidden: false,
            call: () => { /* Handled by REPL/SlashCommandDispatcher */ }
        },
        {
            name: "bug",
            description: "Report a bug in Claude Code",
            type: "prompt",
            userFacingName: () => "bug",
            isEnabled: () => true,
            isHidden: false,
            getPromptForCommand: async () => [{ type: "text", text: "I found a bug. Please help me report it." }]
        }
    ];

    stubs.forEach(stub => {
        // Safe cast as we are providing the essential props for the registry
        commandRegistry.register(stub as CommandDefinition);
    });
}

export default commandRegistry;
