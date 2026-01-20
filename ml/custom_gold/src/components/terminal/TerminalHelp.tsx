
// Logic from chunk_501.ts (Terminal UI Components & Hooks)

import React, { useState, useCallback, useEffect } from 'react';
import { Box, Text } from 'ink';

// --- Suggestion Item (cs5) ---
export function SuggestionItem({ item, isSelected }: any) {
    return (
        <Box flexDirection="row">
            <Text color={isSelected ? "cyan" : undefined} dimColor={!isSelected}>
                {item.displayText}
            </Text>
            {item.description && <Text dimColor> - {item.description}</Text>}
        </Box>
    );
}

// --- Shortcut Help (oW1) ---
export function TerminalHelp() {
    return (
        <Box flexDirection="row" gap={2}>
            <Box flexDirection="column">
                <Text dimColor>! for bash</Text>
                <Text dimColor>/ for commands</Text>
            </Box>
            <Box flexDirection="column">
                <Text dimColor>ctrl+c to cancel</Text>
                <Text dimColor>ctrl+u to undo</Text>
            </Box>
        </Box>
    );
}

// --- Shell Task Details (rl2) ---
export function ShellTaskDetails({ task }: any) {
    return (
        <Box borderStyle="round" flexDirection="column" paddingX={1}>
            <Text bold>Shell Details</Text>
            <Text>Status: {task.status}</Text>
            <Text>Command: {task.command}</Text>
            <Box borderStyle="single" borderDimColor paddingX={1} height={10}>
                <Text>{task.output || "No output"}</Text>
            </Box>
        </Box>
    );
}

// --- Undo Hook (il2) ---
export function useInputUndo(maxSize = 100) {
    const [buffer, setBuffer] = useState<any[]>([]);
    const push = useCallback((state: any) => {
        setBuffer(prev => [...prev.slice(-maxSize), state]);
    }, [maxSize]);
    return { buffer, push };
}
