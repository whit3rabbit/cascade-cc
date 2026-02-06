/**
 * File: src/components/terminal/StatusLine.tsx
 * Role: Persistent status bar (like vim/tmux) at the bottom of the screen.
 */

import React from 'react';
import { Box, Text } from 'ink';
import { useTheme } from '../../services/terminal/ThemeService.js';

export interface StatusLineProps {
    vimMode: 'NORMAL' | 'INSERT';
    vimModeEnabled: boolean;
    model: string;
    isTyping: boolean;
    cwd: string;
    showTasks: boolean;
    showDiff?: boolean;
    showLoop?: boolean;
    showTeams?: boolean;
    usage?: { inputTokens: number; outputTokens: number };
    planMode: boolean;
    acceptEdits?: boolean;
    exitConfirmation?: boolean;
}

export const StatusLine: React.FC<StatusLineProps> = (props) => {
    const {
        vimMode: _vimMode,
        vimModeEnabled: _vimModeEnabled,
        model,
        isTyping,
        cwd: _cwd,
        showTasks,
        showDiff,
        showLoop,
        showTeams,
        usage: _usage = { inputTokens: 0, outputTokens: 0 },
        planMode,
        acceptEdits,
        exitConfirmation
    } = props;
    const theme = useTheme();
    const [verb, setVerb] = React.useState('Thinking');

    React.useEffect(() => {
        if (!isTyping) return;

        let mounted = true;
        const updateVerb = async () => {
            const { CliActivityTracker } = await import('../../services/telemetry/CliActivityTracker.js');
            if (mounted) {
                setVerb(CliActivityTracker.getInstance().getRandomSpinnerVerb());
            }
        };

        // Initial set
        updateVerb();

        const interval = setInterval(updateVerb, 3000);

        return () => {
            mounted = false;
            clearInterval(interval);
        };
    }, [isTyping]);

    return (
        <Box flexDirection="column" width="100%">
            {/* Horizontal Line above Status Line */}
            <Box width="100%">
                <Text color={theme.subtle}>{"─".repeat(process.stdout.columns || 80)}</Text>
            </Box>

            <Box width="100%" paddingX={2} paddingY={0}>
                <Box flexGrow={1}>
                    {exitConfirmation ? (
                        <Text color={theme.error} bold>
                            Press Ctrl+C again to exit
                        </Text>
                    ) : planMode ? (
                        <Box>
                            <Text color={theme.planMode}>⏸ </Text>
                            <Text color={theme.subtle}>plan mode on </Text>
                            <Text color={theme.inactive}>(shift+Tab to cycle)</Text>
                        </Box>
                    ) : acceptEdits ? (
                        <Box>
                            <Text color={theme.success}>⏵⏵ </Text>
                            <Text color={theme.subtle}>accept edits on </Text>
                            <Text color={theme.inactive}>(shift+Tab to cycle)</Text>
                        </Box>
                    ) : (
                        <Box>
                            <Text color={theme.subtle}>
                                {showTasks ? 'tasks ' : showDiff ? 'diff ' : showLoop ? 'loop ' : showTeams ? 'teams ' : 'default mode '}
                            </Text>
                            {!showTasks && !showDiff && !showLoop && !showTeams && (
                                <Text color={theme.inactive}>(shift+Tab to plan)</Text>
                            )}
                        </Box>
                    )}
                </Box>

                {isTyping && (
                    <Box marginRight={2}>
                        <Text color={theme.suggestion}>{verb}...</Text>
                    </Box>
                )}

                <Box>
                    <Text color={theme.inactive}>{model}</Text>
                </Box>
            </Box>
        </Box>
    );
};
