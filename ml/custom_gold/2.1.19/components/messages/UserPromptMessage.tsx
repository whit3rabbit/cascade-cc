/**
 * File: src/components/messages/UserPromptMessage.tsx
 * Role: Input component for user prompts.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';

export interface UserPromptMessageProps {
    onSubmit: (value: string) => void;
    onClear?: () => void;
    history?: string[];
    vimModeEnabled?: boolean;
    onVimModeChange?: (mode: 'NORMAL' | 'INSERT') => void;
}

export const UserPromptMessage: React.FC<UserPromptMessageProps> = (props) => {
    const { onSubmit, onClear, history = [] } = props;
    const [value, setValue] = useState('');
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [isSearching, setIsSearching] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchMatchIndex, setSearchMatchIndex] = useState(-1);

    // Vim Mode State
    const [vimMode, setVimMode] = useState<'NORMAL' | 'INSERT'>('INSERT');
    const [cursorPos, setCursorPos] = useState(0);

    // Effect to reset to INSERT if global vim mode is disabled
    useEffect(() => {
        if (!props.vimModeEnabled) {
            setVimMode('INSERT');
            props.onVimModeChange?.('INSERT');
        } else {
            setVimMode('NORMAL');
            props.onVimModeChange?.('NORMAL');
        }
    }, [props.vimModeEnabled]);

    useInput((input, key) => {
        // Vim Normal Mode Handling
        if (props.vimModeEnabled && vimMode === 'NORMAL') {
            if (input === 'i') {
                setVimMode('INSERT');
                props.onVimModeChange?.('INSERT');
                return;
            }
            if (input === 'a') {
                setVimMode('INSERT');
                props.onVimModeChange?.('INSERT');
                setCursorPos(Math.min(value.length, cursorPos + 1));
                return;
            }
            if (input === 'I') {
                setVimMode('INSERT');
                props.onVimModeChange?.('INSERT');
                setCursorPos(0);
                return;
            }
            if (input === 'A') {
                setVimMode('INSERT');
                props.onVimModeChange?.('INSERT');
                setCursorPos(value.length);
                return;
            }
            if (input === 'h') {
                setCursorPos(Math.max(0, cursorPos - 1));
                return;
            }
            if (input === 'l') {
                setCursorPos(Math.min(value.length, cursorPos + 1));
                return;
            }
            if (input === '0' || input === '^') {
                setCursorPos(0);
                return;
            }
            if (input === '$') {
                setCursorPos(value.length);
                return;
            }
            if (input === 'x') {
                const newValue = value.slice(0, cursorPos) + value.slice(cursorPos + 1);
                setValue(newValue);
                return;
            }
            if (input === 'D') {
                const newValue = value.slice(0, cursorPos);
                setValue(newValue);
                return;
            }

            // Navigate history with k/j in Normal mode
            if (input === 'k') {
                if (historyIndex < history.length - 1) {
                    const newIndex = historyIndex + 1;
                    setHistoryIndex(newIndex);
                    setValue(history[history.length - 1 - newIndex] || '');
                    setCursorPos(0);
                }
                return;
            }
            if (input === 'j') {
                if (historyIndex > -1) {
                    const newIndex = historyIndex - 1;
                    setHistoryIndex(newIndex);
                    if (newIndex === -1) {
                        setValue('');
                    } else {
                        setValue(history[history.length - 1 - newIndex] || '');
                    }
                }
                return;
            }
            return;
        }

        // Global shortcuts (Search, etc)
        if (key.ctrl && input === 'r') {
            if (!isSearching) {
                setIsSearching(true);
                setSearchQuery('');
                setSearchMatchIndex(-1);
            }
            return;
        }

        // Insert Mode / Standard Search Handling
        if (isSearching) {
            if (key.return) {
                setIsSearching(false);
                if (searchMatchIndex !== -1) {
                    setValue(history[history.length - 1 - searchMatchIndex]);
                }
                setSearchMatchIndex(-1);
                return;
            }
            if (key.escape || (key.ctrl && input === 'c')) {
                setIsSearching(false);
                setValue('');
                return;
            }
            if (!key.ctrl && !key.meta && input) {
                const newQuery = searchQuery + input;
                setSearchQuery(newQuery);
                const matchIdx = history.slice().reverse().findIndex(h => h.includes(newQuery));
                setSearchMatchIndex(matchIdx);
            }
            if (key.backspace) {
                const newQuery = searchQuery.slice(0, -1);
                setSearchQuery(newQuery);
                if (newQuery === '') {
                    setSearchMatchIndex(-1);
                } else {
                    const matchIdx = history.slice().reverse().findIndex(h => h.includes(newQuery));
                    setSearchMatchIndex(matchIdx);
                }
            }
            return;
        }

        // Standard Navigation / Editing (Insert Mode)
        if (key.escape && props.vimModeEnabled) {
            setVimMode('NORMAL');
            props.onVimModeChange?.('NORMAL');
            return;
        }

        if (key.upArrow) {
            if (historyIndex < history.length - 1) {
                const newIndex = historyIndex + 1;
                setHistoryIndex(newIndex);
                setValue(history[history.length - 1 - newIndex] || '');
            }
        }
        if (key.downArrow) {
            if (historyIndex > -1) {
                const newIndex = historyIndex - 1;
                setHistoryIndex(newIndex);
                if (newIndex === -1) {
                    setValue('');
                } else {
                    setValue(history[history.length - 1 - newIndex] || '');
                }
            }
        }
        if (input === 'l' && key.ctrl) {
            onClear?.();
        }
    });

    if (isSearching) {
        return (
            <Box>
                <Text color="yellow">(reverse-i-search)`{searchQuery}`': </Text>
                <Text>{searchMatchIndex !== -1 ? history[history.length - 1 - searchMatchIndex] : ''}</Text>
            </Box>
        );
    }

    // Custom rendering for Vim Cursor
    if (props.vimModeEnabled && vimMode === 'NORMAL') {
        return (
            <Box>
                <Box marginRight={1}>
                    <Text color="green" bold>[N]</Text>
                </Box>
                <Text>
                    {value.slice(0, cursorPos)}
                    <Text backgroundColor="green" color="black">{value[cursorPos] || ' '}</Text>
                    {value.slice(cursorPos + 1)}
                </Text>
            </Box>
        );
    }

    return (
        <Box>
            <Box marginRight={1}>
                <Text color="cyan">{'>'}</Text>
            </Box>
            <TextInput
                value={value}
                onChange={(val) => {
                    setValue(val);
                    setHistoryIndex(-1);
                    setCursorPos(val.length);
                }}
                onSubmit={(val) => {
                    setValue('');
                    setHistoryIndex(-1);
                    onSubmit(val);
                }}
            />
        </Box>
    );
};
