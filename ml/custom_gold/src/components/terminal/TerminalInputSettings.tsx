
// Logic from chunk_512.ts (Input Interactivity & Settings)

import React, { useState } from 'react';
import { Box, Text } from 'ink';

// --- Prompt Prefix (d17) ---
export function PromptPrefix({ agentName, isLoading }: any) {
    const prefix = agentName ? `[${agentName}]` : "";
    return (
        <Text dimColor={isLoading}>
            <Text color="cyan">{prefix}</Text>
            <Text color="white">&gt; </Text>
        </Text>
    );
}

// --- Model Picker (jDA) ---
export function ModelPicker({ onSelect, onCancel }: any) {
    const models = [
        { label: "Claude 3.5 Sonnet", value: "claude-3-5-sonnet-20241022" },
        { label: "Claude 3 Opus", value: "claude-3-opus-20240229" }
    ];
    return (
        <Box flexDirection="column" borderStyle="round" paddingX={1}>
            <Text bold color="remember">Select Model</Text>
            {models.map((m, i) => (
                <Text key={m.value}>{i + 1}. {m.label}</Text>
            ))}
            <Box marginTop={1}><Text dimColor>Use arrows and enter to select</Text></Box>
        </Box>
    );
}

// --- Stash Indicator (Xt2) ---
export function StashIndicator({ hasStash }: { hasStash: boolean }) {
    if (!hasStash) return null;
    return (
        <Box paddingLeft={2}>
            <Text dimColor italic>→ Stashed (restores after submit)</Text>
        </Box>
    );
}
