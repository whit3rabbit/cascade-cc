
import React from 'react';
import { Box, Text } from 'ink';
import { renderMarkdownToTerminal } from '../../utils/terminal/TerminalMarkdown.js';

// Based on chunk_477.ts:149-154
export const Markdown: React.FC<{ children: string }> = ({ children }) => {
    // Assuming useTheme and useSettings are available
    const syntaxHighlightingDisabled = false; // Mocked
    return (
        <Text>
            {renderMarkdownToTerminal(children, {}, syntaxHighlightingDisabled)}
        </Text>
    );
};

// Based on chunk_477.ts:163-179
export const RejectedPlanDisplay: React.FC<{ plan: string }> = ({ plan }) => {
    return (
        <Box flexDirection="column" marginTop={1}>
            <Text color="red">User rejected Claude's plan:</Text>
            <Box borderStyle="round" borderColor="red" paddingX={1}>
                <Text dimColor>
                    <Markdown>{plan}</Markdown>
                </Text>
            </Box>
        </Box>
    );
};

// Based on chunk_477.ts:196-238
export const PlanStatusMessage: React.FC<{
    plan: string;
    filePath?: string;
    awaitingLeaderApproval?: boolean;
    approved?: boolean
}> = ({ plan, filePath, awaitingLeaderApproval, approved }) => {
    const isPlanEmpty = !plan || plan.trim() === "";
    const displayPath = filePath ? filePath : ""; // Simplified for now

    if (isPlanEmpty) {
        return (
            <Box flexDirection="column" marginTop={1}>
                <Box flexDirection="row">
                    <Text color="cyan">✓</Text>
                    <Text> Exited plan mode</Text>
                </Box>
            </Box>
        );
    }

    if (awaitingLeaderApproval) {
        return (
            <Box flexDirection="column" marginTop={1}>
                <Box flexDirection="row">
                    <Text color="cyan">✓</Text>
                    <Text> Plan submitted for team lead approval</Text>
                </Box>
                <Box flexDirection="column" marginLeft={2}>
                    {filePath && <Text dimColor>Plan file: {displayPath}</Text>}
                    <Text dimColor>Waiting for team lead to review and approve...</Text>
                </Box>
            </Box>
        );
    }

    if (approved) {
        return (
            <Box flexDirection="column" marginTop={1}>
                <Box flexDirection="row">
                    <Text color="cyan">✓</Text>
                    <Text> User approved Claude's plan</Text>
                </Box>
                <Box flexDirection="column" marginLeft={2}>
                    {filePath && <Text dimColor>Plan saved to: {displayPath} · /plan to edit</Text>}
                    <Box marginTop={1}>
                        <Markdown>{plan}</Markdown>
                    </Box>
                </Box>
            </Box>
        );
    }

    return null;
};

// Based on chunk_477.ts:240-251
export const CurrentPlanDisplay: React.FC<{ plan: string }> = ({ plan }) => {
    return (
        <Box flexDirection="column" marginTop={1}>
            <RejectedPlanDisplay plan={plan || "No plan found"} />
        </Box>
    );
};
