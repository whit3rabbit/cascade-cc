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

## Current Version: v2.1.19

### Key Features:
- **Sandbox Refinements**: Improved Linux \`bwrap\` and macOS \`sandbox-exec\` isolation.
- **Enhanced Keychain**: Faster and more robust API key management with in-memory caching.
- **Beta OAuth**: Sign in with your Anthropic account directly from the CLI.
- **Model Switcher**: Easily switch between Sonnet, Opus, and Haiku models using \`/model\`.

### Bug Fixes:
- Fixed an issue where terminal detection failed on some macOS systems.
- Improved error handling for network-restricted environments.
- Corrected token count estimation for large file reads.

### [Full Changelog](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md)

${userInput ? `Focus on: ${userInput}` : ""}
`
            }
        ];
    },
    userFacingName() {
        return "release notes";
    }
});
