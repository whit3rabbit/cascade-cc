/**
 * File: src/commands/init.ts
 * Role: Implementation of the /init command for project initialization.
 */

import { createCommandHelper, CommandContext } from './helpers.js';

/**
 * Standardized /init command definition.
 * Used for project initialization and CLAUDE.md creation.
 */
export const initCommandDefinition = createCommandHelper("init", "Initialize the project configuration and create CLAUDE.md", {
    async getPromptForCommand(userInput: string, _context: CommandContext) {
        return [
            {
                type: "text",
                text: `You are an AI assistant helping the user initialize their project for use with Claude Code.
Your goal is to create or update a CLAUDE.md file in the project root.

Instructions:
1.  **Check for existence**: Determine if CLAUDE.md exists in the current directory.
2.  **Analyze project**: If it doesn't exist, analyze the project structure (e.g., look for package.json, go.mod, etc.) to infer build, test, and lint commands.
3.  **Propose content**: Propose a CLAUDE.md content including:
    -   **Build Commands**: How to build the project.
    -   **Test Commands**: How to run tests.
    -   **Lint Commands**: How to run linting.
    -   **Code Style Guidelines**: Any project-specific style rules.
4.  **Interactive process**: Ask the user to confirm or provide the correct commands if they cannot be inferred.
5.  **Completion**: Once the user is satisfied, write the CLAUDE.md file.

${userInput ? `Additional User Request: ${userInput}` : ""}

Please guide the user through this process in a friendly and helpful manner.`
            }
        ];
    },
    userFacingName() {
        return "init";
    }
});
