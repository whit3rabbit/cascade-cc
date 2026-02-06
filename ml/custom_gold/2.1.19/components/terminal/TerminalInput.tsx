import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { useTerminalInput } from '../../hooks/useTerminalInput.js';

import { VimMode } from '../../hooks/useVimMode.js';

export interface TerminalInputProps {
    value: string;
    onChange: (value: string) => void;
    onSubmit: (value: string) => void;
    onExit: () => void;
    history: string[];
    vimModeEnabled?: boolean;
    onVimModeChange?: (mode: VimMode) => void;
    planMode?: boolean;
    suggestions?: string[];
    onClearScreen?: () => void;
    agentName?: string;
    agentColor?: string;
    commands?: any[];
    agents?: any[];
}

/**
 * TerminalInput equivalent to Rd2 (memoized as lTK) in golden source.
 * This is the actual interactive area where the user types.
 */
export const TerminalInput: React.FC<TerminalInputProps> = (props) => {
    const {
        value,
        onChange,
        onSubmit,
        onExit,
        history,
        vimModeEnabled,
        onVimModeChange,
        planMode,
        agentName,
        agentColor,
        commands,
        agents
    } = props;

    const [cursorOffset, setCursorOffset] = useState(value.length);

    const {
        onInput: terminalInputHandler,
        suggestions,
        selectedSuggestion,
        isMacroRecording,
        currentVimMode,
        stashedValue
    } = useTerminalInput({
        value,
        onChange,
        onSubmit: (val) => {
            onSubmit(val);
            setCursorOffset(0);
        },
        onExit,
        onClearInput: () => onChange(''),
        history,
        commands,
        agents,
        multiline: true,
        cursorOffset,
        setCursorOffset,
        vimModeEnabled,
        onVimModeChange
    });

    useInput((input, key) => {
        terminalInputHandler(input, key);
    });

    const renderPrefix = () => {
        if (planMode) {
            return <Text color="magenta">◈ </Text>;
        }
        if (agentName) {
            return <Text color={agentColor as any || 'cyan'}>❯ </Text>;
        }
        return <Text color="blue">❯ </Text>;
    };

    const getCursorStyle = () => {
        if (!vimModeEnabled) return { backgroundColor: '#ffffff', color: '#000000' };

        switch (currentVimMode) {
            case 'NORMAL':
                return { backgroundColor: '#48BB78', color: '#000000' }; // Chakra green.500
            case 'VISUAL':
                return { backgroundColor: '#ECC94B', color: '#000000' }; // Chakra yellow.400
            case 'INSERT':
            default:
                // For INSERT mode, use a white block if cursor is on a character, 
                // or a slightly different style to indicate beam-like behavior
                return { backgroundColor: '#ffffff', color: '#000000' };
        }
    };

    const cursorStyle = getCursorStyle();

    return (
        <Box flexDirection="column">
            <Box flexDirection="row">
                {renderPrefix()}
                <Box flexGrow={1}>
                    <Text>
                        {value.slice(0, cursorOffset)}
                        <Text {...cursorStyle}>
                            {value[cursorOffset] || ' '}
                        </Text>
                        {value.slice(cursorOffset + 1)}
                    </Text>
                </Box>
                {isMacroRecording && (
                    <Box marginLeft={1}>
                        <Text color="red">● recording</Text>
                    </Box>
                )}
                {stashedValue && !value && (
                    <Box marginLeft={1}>
                        <Text color="dimColor">stashed: {stashedValue.length > 10 ? stashedValue.slice(0, 10) + '...' : stashedValue} (Ctrl+S to restore)</Text>
                    </Box>
                )}
            </Box>

            {/* Vim Mode Indicator */}
            {vimModeEnabled && currentVimMode !== 'NORMAL' && (
                <Box marginTop={0}>
                    <Text dimColor>-- {currentVimMode} --</Text>
                </Box>
            )}

            {suggestions.length > 0 && (
                <Box flexDirection="column" marginLeft={2} marginTop={1} borderStyle="round" borderColor="gray">
                    {suggestions.map((s: any) => (
                        <Box key={s.id} backgroundColor={s === selectedSuggestion ? 'blue' : undefined}>
                            <Text color={s === selectedSuggestion ? 'white' : 'gray'}>
                                {s === selectedSuggestion ? ' > ' : '   '}
                                {s.displayText} {s.description && `- ${s.description}`}
                            </Text>
                        </Box>
                    ))}
                </Box>
            )}
        </Box>
    );
};
