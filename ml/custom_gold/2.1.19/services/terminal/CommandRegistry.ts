/**
 * File: src/services/terminal/CommandRegistry.ts
 * Role: Central registry for all CLI commands and slash commands.
 */

import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { CommandDefinition } from '../../commands/helpers.js';
import { helpCommandDefinition, mobileCommandDefinition } from '../../commands/help.js';
import { initCommandDefinition } from '../../commands/init.js';
import { prCommentsCommandDefinition } from '../../commands/pr_comments.js';
import { doctorCommandDefinition } from '../../commands/doctor.js';
import { bugCommandDefinition } from '../../commands/bug.js';

const execAsync = promisify(exec);

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
     * Prioritizes tools/agents if many unstaged changes are detected in CWD.
     */
    async getFilteredCommands(filter: string = ""): Promise<CommandDefinition[]> {
        const all = this.getAllCommands();
        let filtered = all.filter(cmd =>
            cmd.name.includes(filter) ||
            (cmd.description && cmd.description.includes(filter))
        );

        // Git Change Tracking: Check for unstaged changes
        if (filter === "") {
            try {
                const { stdout } = await execAsync('git status --porcelain', { cwd: process.cwd() });
                const lineCount = stdout.split('\n').filter(line => line.trim().length > 0).length;

                if (lineCount > 5) {
                    // Prioritize git-related commands or specific agents if many changes
                    // For now, we hoist 'pr-comments' or similar if they exist/match criteria
                    filtered.sort((a, b) => {
                        const aIsGit = a.name.includes('pr-') || a.name.includes('git');
                        const bIsGit = b.name.includes('pr-') || b.name.includes('git');
                        if (aIsGit && !bIsGit) return -1;
                        if (!aIsGit && bIsGit) return 1;
                        return 0;
                    });
                }
            } catch (e) {
                // Ignore git errors (not a repo, etc)
            }
        }

        return filtered;
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
    commandRegistry.register(prCommentsCommandDefinition as CommandDefinition);
    commandRegistry.register(doctorCommandDefinition as CommandDefinition);
    commandRegistry.register(bugCommandDefinition as CommandDefinition);

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
        }
    ];

    stubs.forEach(stub => {
        // Safe cast as we are providing the essential props for the registry
        commandRegistry.register(stub as CommandDefinition);
    });
}

export default commandRegistry;
