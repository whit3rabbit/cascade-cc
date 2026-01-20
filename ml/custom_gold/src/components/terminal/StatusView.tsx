
// Logic from chunk_540.ts (Status & Theme UI)

import React from 'react';
import { Box, Text } from 'ink';

// --- System Status View (Z39) ---
export function StatusView({ appState, context }: any) {
    return (
        <Box flexDirection="column" marginTop={1}>
            <Box flexDirection="row" gap={1}>
                <Text bold>Model:</Text>
                <Text>{appState.mainLoopModel || "Claude 3.5 Sonnet"}</Text>
            </Box>
            <Box flexDirection="row" gap={1}>
                <Text bold>Version:</Text>
                <Text>2.0.76</Text>
            </Box>
            {/* ... Auth, MCP, Memory status ... */}
            <Box flexDirection="column" marginTop={1}>
                <Text bold>System Diagnostics</Text>
                {/* Warnings list */}
            </Box>
        </Box>
    );
}

// --- Theme Selector (oDA) ---
export function ThemeSelector({ onSelect }: any) {
    return (
        <Box flexDirection="column" marginX={1}>
            <Text bold color="permission">Choose the text style that looks best with your terminal</Text>
            <Box flexDirection="column" marginY={1}>
                <Text>Dark mode</Text>
                <Text>Light mode</Text>
                <Text>ANSI colors only</Text>
            </Box>
            {/* Live Preview Placeholder */}
            <Box borderStyle="single" borderColor="subtle" paddingX={1}>
                <Text color="cyan">function greet() {'{'}</Text>
                <Text color="green">  console.log("Hello, Claude!");</Text>
                <Text color="cyan">{'}'}</Text>
            </Box>
        </Box>
    );
}

// --- Modal Header (I6) ---
export function ModalHeader({ title, subtitle, onCancel }: any) {
    return (
        <Box flexDirection="column" gap={1} marginBottom={1}>
            <Box justifyContent="space-between">
                <Text bold color="permission">{title}</Text>
                <Text dimColor>Esc to close</Text>
            </Box>
            {subtitle && <Text dimColor>{subtitle}</Text>}
        </Box>
    );
}
