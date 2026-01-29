/**
 * File: src/services/terminal/CommandRegistry.ts
 * Role: Central registry for all CLI commands and slash commands.
 */

interface Command {
    name: string;
    description?: string;
    aliases?: string[];
    [key: string]: any;
}

export class CommandRegistry {
    private commands: Map<string, Command>;
    private aliases: Map<string, string>;

    constructor() {
        this.commands = new Map();
        this.aliases = new Map();
    }

    /**
     * Registers a command object.
     */
    register(command: Command): void {
        this.commands.set(command.name, command);
        if (command.aliases) {
            command.aliases.forEach(alias => {
                this.aliases.set(alias, command.name);
            });
        }
    }

    /**
     * Finds a command by name or alias.
     */
    findCommand(name: string): Command | null | undefined {
        if (this.commands.has(name)) return this.commands.get(name);
        const realName = this.aliases.get(name);
        if (realName) return this.commands.get(realName);
        return null;
    }

    /**
     * Returns all registered commands.
     */
    getAllCommands(): Command[] {
        return Array.from(this.commands.values());
    }

    /**
     * Returns commands filtered by eligibility and search term.
     */
    async getFilteredCommands(filter: string = ""): Promise<Command[]> {
        const all = this.getAllCommands();
        return all.filter(cmd =>
            cmd.name.includes(filter) ||
            (cmd.description && cmd.description.includes(filter))
        );
    }
}

/**
 * Initializes and registers all built-in commands.
 */
export async function initializeCommandDefinitions(): Promise<void> {
    // In a full implementation, this would import each command and register it.
    console.log("Initializing commands...");
}

export const commandRegistry = new CommandRegistry();
export default commandRegistry;
