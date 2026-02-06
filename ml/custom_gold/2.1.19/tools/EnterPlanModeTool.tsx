/**
 * File: src/tools/EnterPlanModeTool.tsx
 * Role: Tool for requesting permission to enter planning mode.
 */

import React from 'react';
import { Text, Box } from 'ink';

export const ENTER_PLAN_MODE_DESCRIPTION = `Use this tool proactively when you're about to start a non-trivial implementation task. Getting user sign-off on your approach before writing code prevents wasted effort and ensures alignment. This tool transitions you into plan mode where you can explore the codebase and design an implementation approach for user approval.

## When to Use This Tool
Prefer using EnterPlanMode for implementation tasks unless they're simple.

1. **New Feature Implementation**: Adding meaningful new functionality.
2. **Multiple Valid Approaches**: The task can be solved in several different ways.
3. **Multi-File Changes**: The task will likely touch more than 2-3 files.
`;

export interface EnterPlanModeResult {
    message: string;
}

/**
 * Result component for EnterPlanMode tool.
 */
export const EnterPlanModeResultView: React.FC<{ data: EnterPlanModeResult }> = () => (
    <Box flexDirection="column" marginTop={1}>
        <Box flexDirection="row">
            <Text color="green">✅</Text>
            <Text> Entered plan mode</Text>
        </Box>
        <Box paddingLeft={2}>
            <Text dimColor>Claude is now exploring and designing an implementation approach.</Text>
        </Box>
    </Box>
);

/**
 * Rejected component for EnterPlanMode tool.
 */
export const EnterPlanModeRejectedView: React.FC = () => (
    <Box flexDirection="row" marginTop={1}>
        <Text color="red">❌</Text>
        <Text> User declined to enter plan mode</Text>
    </Box>
);

// Tool definition stub
export const EnterPlanModeTool = {
    name: "EnterPlanMode",
    description: "Requests permission to enter plan mode for complex tasks requiring exploration and design.",
    prompt: ENTER_PLAN_MODE_DESCRIPTION,
    inputSchema: {},
    outputSchema: {
        message: { type: "string" }
    },
    async call(input: any, context: any): Promise<{ data: EnterPlanModeResult }> {
        if (context && typeof context.setPlanMode === 'function') {
            context.setPlanMode(true);
        }
        return {
            data: {
                message: "Entered plan mode. You should now focus on exploring the codebase."
            }
        };
    }
};
