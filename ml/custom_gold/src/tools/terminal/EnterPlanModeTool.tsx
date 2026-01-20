
import React from 'react';
import { z } from 'zod';
import { Box, Text } from 'ink';

// EnterPlanMode tool (vvA)
export const EnterPlanModeTool = {
    name: "EnterPlanMode",
    async description() {
        return "Requests permission to enter plan mode for complex tasks requiring exploration and design";
    },
    async prompt() {
        // Detailed prompt from Dd2 (chunk_487:196)
        return `Use this tool proactively when you're about to start a non-trivial implementation task. Getting user sign-off on your approach before writing code prevents wasted effort and ensures alignment. This tool transitions you into plan mode where you can explore the codebase and design an implementation approach for user approval.

## When to Use This Tool
... (truncated for brevity in deobfuscation, but ideally would include full text)
`;
    },
    inputSchema: z.object({}).strict(),
    outputSchema: z.object({
        message: z.string().describe("Confirmation that plan mode was entered")
    }),

    userFacingName() { return ""; },
    isEnabled() { return true; },
    isConcurrencySafe() { return true; },
    isReadOnly() { return true; },

    async checkPermissions(input: any) {
        return {
            behavior: "ask",
            message: "Enter plan mode?",
            updatedInput: input
        };
    },

    async call(input: any, context: any) {
        if (context.agentId) throw Error("EnterPlanMode tool cannot be used in agent contexts");

        // This tool updates the global session state to 'plan' mode
        // In a real implementation, this would call context.setAppState

        return {
            data: {
                message: "Entered plan mode. You should now focus on exploring the codebase and designing an implementation approach."
            }
        };
    },

    mapToolResultToToolResultBlockParam({ message }: { message: string }, toolUseId: string) {
        return {
            type: "tool_result",
            tool_use_id: toolUseId,
            content: `${message}\n\nIn plan mode, you should:\n1. Thoroughly explore the codebase\n2. Identify similar features\n3. Consider trade-offs\n4. Use AskUserQuestion if needed\n5. Design strategy\n6. Use ExitPlanMode when ready\n\nRemember: DO NOT write or edit any files yet.`
        };
    }
};

export default EnterPlanModeTool;
