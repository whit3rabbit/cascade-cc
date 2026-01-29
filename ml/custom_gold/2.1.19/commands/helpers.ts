/**
 * File: src/commands/helpers.ts
 * Role: Command Helper Functions and Interfaces
 */

export interface CommandContext {
    cwd: string;
    version: string;
    [key: string]: any;
}

export interface CommandDefinition {
    type: string;
    name: string;
    description: string;
    progressMessage: string;
    isEnabled: () => boolean;
    isHidden: boolean;
    userFacingName: () => string;
    source: string;
    getPromptForCommand: (userInput: string, context: CommandContext) => Promise<any[]>;
    [key: string]: any;
}

/**
 * Utility to create a standardized command definition.
 * 
 * @param {string} commandName - The name of the command (e.g., 'init').
 * @param {string} description - Brief description of what the command does.
 * @param {Partial<CommandDefinition>} [additionalProps={}] - Extra properties to override defaults.
 * @returns {CommandDefinition} A standardized command definition object.
 */
export function createCommandHelper(
    commandName: string,
    description: string,
    additionalProps: Partial<CommandDefinition> = {}
): CommandDefinition {
    return {
        type: "prompt",
        name: commandName,
        description,
        progressMessage: `Processing ${commandName}...`,
        isEnabled: () => true,
        isHidden: false,
        userFacingName() {
            return commandName;
        },
        source: "builtin",
        async getPromptForCommand(userInput: string, context: CommandContext) {
            if (additionalProps.getPromptForCommand) {
                return additionalProps.getPromptForCommand(userInput, context);
            }
            return [];
        },
        ...additionalProps
    };
}
