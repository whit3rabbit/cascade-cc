
// Logic from chunk_542.ts (Usage & Context UI)

import React from 'react';
import { Box, Text } from 'ink';

// --- High-res Progress Bar (NbA) ---
const BLOCKS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
export function ProgressBar({ ratio, width }: { ratio: number, width: number }) {
    const filled = Math.floor(ratio * width);
    return (
        <Text color="rate_limit_fill" backgroundColor="rate_limit_empty">
            {BLOCKS[BLOCKS.length - 1].repeat(filled)}
            {ratio < 1 ? BLOCKS[0].repeat(width - filled) : ""}
        </Text>
    );
}

// --- Usage Panel (D39) ---
export function UsagePanel({ usageData }: any) {
    if (!usageData) return <Text dimColor>Loading usage data...</Text>;

    return (
        <Box flexDirection="column" marginTop={1} gap={1}>
            <Text bold>Current session</Text>
            <Box flexDirection="row" gap={1}>
                <ProgressBar ratio={usageData.sessionUtilization / 100} width={50} />
                <Text>{usageData.sessionUtilization}% used</Text>
            </Box>
            <Text dimColor>Resets at {new Date(usageData.sessionReset).toLocaleTimeString()}</Text>
        </Box>
    );
}

// --- Context usage visualization ($39) ---
export function ContextUsage({ data }: any) {
    return (
        <Box flexDirection="column" padding={1}>
            <Text bold>Context Usage</Text>
            <Box flexDirection="row" gap={2}>
                {/* 10x10 Grid Simulation */}
                <Box flexDirection="column">
                    <Text>⛁ ⛁ ⛁ ⛶ ⛶</Text>
                </Box>
                <Box flexDirection="column">
                    <Text dimColor>{data.totalTokens / 1000}k / {data.maxTokens / 1000}k tokens</Text>
                </Box>
            </Box>
        </Box>
    );
}
