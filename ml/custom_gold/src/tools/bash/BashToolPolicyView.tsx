
// Logic from chunk_526.ts (Tool Feedback & Bash Policy)

import React from 'react';
import { Box, Text } from 'ink';

// --- Tool Progress View (PV1) ---
export function ToolExecutionProgress({ tools, verbose, terminalSize }: any) {
    if (!tools.length) return <Text dimColor > Initializing…</Text>;

    // Condensed view if terminal is too small
    const isSmall = terminalSize && terminalSize.rows < 10;
    if (isSmall) {
        return (
            <Text dimColor >
            In progress… · <Text bold > { tools.length } </Text> tool uses
                </Text>
        );
    }

    return (
        <Box flexDirection= "column" >
        {
            tools.map((t: any, i: number) => (
                <Text key= { i } > Executing { t.name }...</Text>
            ))
        }
        </Box>
    );
}

// --- Bash Policy Prompt (_B7 / f99) ---
export function getBashToolPrompt(config: any) {
    const sandboxPrompt = config.sandboxingEnabled ? `
- Commands run in a sandbox by default. 
- Set dangerouslyDisableSandbox: true only if explicitly requested or if access is denied.
` : "Sandbox is disabled by policy.";

    return `Executes a bash command.
${sandboxPrompt}
- Quoting: Always quote paths with spaces.
- Attribution: Commits will include "Generated with Claude Code".
- Absolute Paths: Prefer absolute paths over 'cd'.
`;
}

export function getGitAttributionFooter(user: string) {
    return `\n\nCo-Authored-By: ${user} <noreply@anthropic.com>\nGenerated with Claude Code`;
}
