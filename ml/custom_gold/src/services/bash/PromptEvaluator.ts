import { BashTool } from "../../tools/bash/BashTool.js";
import { log } from "../logger/loggerService.js";

const logger = log("prompt-evaluator");

// Regex from chunk_342.ts (_31)
const CODE_BLOCK_REGEX = /```!\s*\n?([\s\S]*?)\n?```/g;
const INLINE_REGEX = /(?<!\w|\$)!`([^`]+)`/g;

/**
 * Evaluates bash commands embedded in prompts.
 * Deobfuscated from PWA in chunk_342.ts.
 * 
 * It looks for commands in ```! command ``` or !`command` format and replaces 
 * them with their execution output.
 */
export async function evaluatePromptBashCommands(prompt: string, context: any, source: string): Promise<string> {
    let result = prompt;

    // Find all matches for both regexes
    const codeBlockMatches = [...prompt.matchAll(CODE_BLOCK_REGEX)];
    const inlineMatches = [...prompt.matchAll(INLINE_REGEX)];
    const allMatches = [...codeBlockMatches, ...inlineMatches];

    if (allMatches.length === 0) return prompt;

    // Process matches
    for (const match of allMatches) {
        const command = match[1]?.trim();
        if (!command) continue;

        try {
            // Execute the command using BashTool
            // We pass context which should include what's needed for permission checks
            const bashResult = await BashTool.call({ command }, context);

            // Format output (approximating CQ2 logic from chunk_342.ts)
            const stdout = (bashResult.data.stdout || "").trim();
            const stderr = (bashResult.data.stderr || "").trim();

            let output = stdout;
            if (stderr) {
                if (output) {
                    output += `\n[stderr]\n${stderr}`;
                } else {
                    output = `[stderr]\n${stderr}`;
                }
            }

            // Replace the original match with command output
            result = result.replace(match[0], output);
        } catch (error: any) {
            // Handle errors (approximating gA5 logic from chunk_342.ts)
            logger.error(`Bash command evaluation failed in prompt ${source}: ${command}`, error);

            const errorMsg = error instanceof Error ? error.message : String(error);
            const formattedError = `[Error]\n${errorMsg}`;

            result = result.replace(match[0], formattedError);
        }
    }

    return result;
}
