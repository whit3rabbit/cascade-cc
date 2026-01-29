/**
 * File: src/commands/release-notes.ts
 * Role: Implementation of the /release-notes command.
 */

import { createCommandHelper, CommandContext } from './helpers.js';

/**
 * Command definition for displaying release notes and version updates.
 */
export const releaseNotesCommandDefinition = createCommandHelper("release-notes", "Display the latest release notes and updates", {
    async getPromptForCommand(userInput: string, _context: CommandContext) {
        return [
            {
                type: "text",
                text: `You are an AI assistant for the Claude Code CLI tool. Your task is to provide the user with the latest release notes and version history.

Instructions:
1.  **Locate Release Info**: Check for a CHANGELOG.md file or use git tags to identify recent changes.
2.  **Summarize**: Provide a concise summary of the latest version (v2.1.19), including key features, bug fixes, and breaking changes.
3.  **Future Updates**: If applicable, mention any upcoming features or known issues currently being addressed.

If no local release notes are found, provide a summary of general Claude Code updates available online.

${userInput ? `Focus on: ${userInput}` : ""}
`
            }
        ];
    },
    userFacingName() {
        return "release-notes";
    }
});
