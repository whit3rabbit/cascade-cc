
// Logic from chunk_520.ts (Tips & Network Permissions)

import React, { useState } from 'react';
import { Box, Text } from 'ink';

// --- Pro Tips Service (DA9) ---
export const TIPS = [
    { id: "plan-mode", content: "Use Plan Mode for complex tasks", cooldown: 5 },
    { id: "memory", content: "Run /memory to manage Claude's memory", cooldown: 10 }
];

export function getNextTip(history: any) {
    // Filter relevant tips based on user usage and cooldowns
    return TIPS[0];
}

// --- Network Permission Prompt (dw0) ---
export function NetworkPermissionPrompt({ host, onResponse }: any) {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor="warning" paddingX={1}>
            <Text bold>Network access requested: {host}</Text>
            <Text dimColor>Do you want to allow this external connection?</Text>

            <Box marginTop={1} flexDirection="column">
                <Text>1. Yes</Text>
                <Text>2. Always Allow for this host</Text>
                <Text>3. No (Esc)</Text>
            </Box>
        </Box>
    );
}

// --- Date Utils (Inlined logic) ---
export const DateUtils = {
    isDateValid: (d: any) => !isNaN(Date.parse(d)),
    diffDays: (d1: Date, d2: Date) => Math.floor((d2.getTime() - d1.getTime()) / 86400000)
};
