/**
 * File: src/components/terminal/StatusLine.tsx
 * Role: Persistent status bar (like vim/tmux) at the bottom of the screen.
 */

import React from 'react';
import { Box, Text } from 'ink';

export interface StatusLineProps {
    vimMode: 'NORMAL' | 'INSERT';
    vimModeEnabled: boolean;
    model: string;
    isTyping: boolean;
    cwd: string;
    showTasks: boolean;
}

export const StatusLine: React.FC<StatusLineProps> = ({
    vimMode,
    vimModeEnabled,
    model,
    isTyping,
    cwd,
    showTasks
}) => {
    return (
        <Box
            width="100%"
            paddingX={1}
            borderStyle="single"
            borderTop={false}
            borderLeft={false}
            borderRight={false}
            borderBottom={false}
            borderColor="gray"
            backgroundColor="blue" // Premium feel
        >
            <Box width="20%">
                <Text color="black" bold>
                    {vimModeEnabled ? ` ${vimMode} ` : ' STANDARD '}
                </Text>
            </Box>

            <Box width="40%" justifyContent="center">
                <Text color="white">
                    {isTyping ? 'Thinking...' : 'Idle'} | {model}
                </Text>
            </Box>

            <Box width="40%" justifyContent="flex-end">
                <Text color="white">
                    {cwd.split('/').pop()} {showTasks ? '[TASKS]' : ''}
                </Text>
            </Box>
        </Box>
    );
};
