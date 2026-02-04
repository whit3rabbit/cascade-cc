/**
 * File: src/commands/bug.ts
 * Role: Implementation of the /bug (feedback) command.
 */

import { CommandDefinition, createCommandHelper } from './helpers.js';

export const bugCommandDefinition: CommandDefinition = createCommandHelper(
    "bug",
    "Submit feedback or report a bug about Claude Code",
    {
        aliases: ["feedback"],
        type: "prompt",
        isEnabled: () => !process.env.DISABLE_BUG_COMMAND,
        async getPromptForCommand(_args: string, _context: any) {
            return [
                {
                    type: "text",
                    text: "I want to report a bug or provide feedback. Please help me collect the necessary information and I'll submit it to the Anthropic team. (Note: For security, do not include sensitive credentials)."
                }
            ];
        },
        async call(onDone: (result: string) => void) {
            console.log("Feedback command initiated. Our agent will help you draft the report.");
            onDone("Feedback session started");
        }
    }
);
