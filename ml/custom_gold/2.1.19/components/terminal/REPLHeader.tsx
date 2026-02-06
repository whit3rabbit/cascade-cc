import React from 'react';
import { Box, Text } from 'ink';
import { Logo } from './Logo.js';

export interface REPLHeaderProps {
    screen: 'transcript' | 'prompt'; // simplified from golden source 'transcript' | 'prompt'
    setScreen: (screen: 'transcript' | 'prompt') => void;
    setScreenToggleId: (id: number) => void;
    setShowAllInTranscript: (show: boolean) => void;
    onEnterTranscript: () => void;
    onExitTranscript: () => void;
    todos: any[];
    agentName?: string;
    agentColor?: string;
}

/**
 * REPLHeader equivalent to HF6 in golden source.
 * Handles the top-level status and logo display.
 */
export const REPLHeader: React.FC<REPLHeaderProps> = ({
    screen: _screen,
    agentName,
    agentColor
}) => {
    // Current REPL.tsx shows Logo only if messages.length === 0.
    // Golden source HF6 seems to handle more persistent status.

    return (
        <Box flexDirection="column" width="100%" marginBottom={1}>
            <Logo
                version="2.1.27"
                model="Sonnet 4.5"
                cwd={process.cwd()}
                subscription="Pro" // Placeholder
            />
            {agentName && (
                <Box marginTop={1}>
                    <Text>Active Agent: </Text>
                    <Text color={agentColor as any || 'cyan'}>{agentName}</Text>
                </Box>
            )}
        </Box>
    );
};
