/**
 * File: src/components/messages/UserPromptMessage.tsx
 * Role: Input component for user prompts.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { useTheme } from '../../services/terminal/ThemeService.js';
import { spawnSync } from 'child_process';
import { writeFileSync, readFileSync, unlinkSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import Fuse from 'fuse.js';

export interface UserPromptMessageProps {
    onSubmit: (value: string) => void;
    onClear?: () => void;
    history?: string[];
    vimModeEnabled?: boolean;
    onVimModeChange?: (mode: 'NORMAL' | 'INSERT') => void;
    planMode?: boolean;
    suggestions?: string[];
    onCancel?: () => void;
}

export const UserPromptMessage: React.FC<UserPromptMessageProps> = (props) => {
    const { onSubmit, onClear, history = [], planMode = false } = props;
    const theme = useTheme();
    const [value, setValue] = useState('');
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [isSearching, setIsSearching] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchMatchIndex, setSearchMatchIndex] = useState(-1);
    const [suggestion, setSuggestion] = useState('');

    // Vim Mode State
    const [vimMode, setVimMode] = useState<'NORMAL' | 'INSERT' | 'COMMAND' | 'VISUAL'>('INSERT');
    const [cursorPos, setCursorPos] = useState(0);
    const [undoStack, setUndoStack] = useState<string[]>([]);
    const [commandValue, setCommandValue] = useState('');

    // Emacs-style Kill Ring
    const [killRing, setKillRing] = useState<string[]>([]);
    const [yankPointer, setYankPointer] = useState(0);

    // Fuse instances for suggestions
    const fuseSuggestions = useMemo(() => new Fuse(props.suggestions || [], { threshold: 0.4 }), [props.suggestions]);
    const fuseHistory = useMemo(() => new Fuse(history, { threshold: 0.4 }), [history]);

    // Effect to update suggestion based on input (Fuzzyish)
    useEffect(() => {
        if (value && props.suggestions) {
            const results = fuseSuggestions.search(value);
            if (results.length > 0) {
                const match = results[0].item;
                if (match.startsWith(value)) {
                    setSuggestion(match.slice(value.length));
                } else {
                    setSuggestion('');
                }
            } else {
                setSuggestion('');
            }
        } else {
            setSuggestion('');
        }
    }, [value, fuseSuggestions]);

    // Simple fuzzy matcher for reverse-i-search
    const getFuzzyMatch = (query: string, list: string[]): number => {
        if (!query) return -1;
        const results = fuseHistory.search(query);
        if (results.length > 0) {
            // Find the item in the list and get its index from the end
            const item = results[0].item;
            return list.slice().reverse().indexOf(item);
        }
        return -1;
    };

    useInput((input, key) => {
        // Multi-line Editing (Shift + Enter or Option + Enter)
        if ((key.return && key.shift) || (key.return && key.meta)) {
            setValue(prev => prev + '\n');
            return;
        }

        // Ctrl+G: Open in Editor
        if (key.ctrl && input === 'g') {
            try {
                const editor = process.env.EDITOR || 'vim';
                const tempFile = join(tmpdir(), `claude-prompt-${Date.now()}.txt`);
                writeFileSync(tempFile, value);

                // We need to be careful with 'spawn' in ink as it might mess with the TUI.
                // However, many ink apps do this by suspending or just spawning sync.
                // spawnSync with { stdio: 'inherit' } is common for editors.
                spawnSync(editor, [tempFile], { stdio: 'inherit' });

                const newValue = readFileSync(tempFile, 'utf8');
                setValue(newValue);
                setCursorPos(newValue.length);
                unlinkSync(tempFile);
            } catch (err) {
                // Fail silently or log? Let's just avoid crashing.
            }
            return;
        }

        // Vim Normal Mode Handling (Enhanced)
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
            if (input === 'w') {
                const nextSpace = value.indexOf(' ', cursorPos);
                setCursorPos(nextSpace === -1 ? value.length : nextSpace + 1);
                return;
            }
            if (input === 'b') {
                const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
                setCursorPos(prevSpace === -1 ? 0 : prevSpace + 1);
                return;
            }
            if (input === 'u') {
                if (undoStack.length > 0) {
                    const last = undoStack[undoStack.length - 1];
                    setUndoStack(prev => prev.slice(0, -1));
                    setValue(last);
                }
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
            if (input === 'x') {
                const newValue = value.slice(0, cursorPos) + value.slice(cursorPos + 1);
                setValue(newValue);
                return;
            }
            if (input === ':') {
                setVimMode('COMMAND');
                setCommandValue('');
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
            if (input === '0' || input === '^') {
                setCursorPos(0);
                return;
            }
            if (input === '$') {
                setCursorPos(value.length);
                return;
            }
            if (input === 'g') {
                // gg detection would require state, assuming G for now as per docs stub for simplicity in one-shot
                // or just handle 'G' below.
            }
            if (input === 'G') {
                setCursorPos(value.length); // End of input
                return;
            }

            // Editing
            if (input === 'd') {
                // dd delete line
                setUndoStack(prev => [...prev.slice(-49), value]);
                setValue('');
                setCursorPos(0);
                return;
            }
            if (input === 'D') {
                const newValue = value.slice(0, cursorPos);
                setValue(newValue);
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
            } else {
                // Continue searching for next match
                const nextIdx = history.slice().reverse().slice(searchMatchIndex + 1).findIndex(h => h.includes(searchQuery));
                if (nextIdx !== -1) setSearchMatchIndex(searchMatchIndex + 1 + nextIdx);
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
                return;
            }
            if (!key.ctrl && !key.meta && input) {
                const newQuery = searchQuery + input;
                setSearchQuery(newQuery);
                setSearchMatchIndex(getFuzzyMatch(newQuery, history));
            }
            if (key.backspace) {
                const newQuery = searchQuery.slice(0, -1);
                setSearchQuery(newQuery);
                setSearchMatchIndex(getFuzzyMatch(newQuery, history));
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

        // macOS style word navigation (Option + Arrow)
        if (key.meta && key.leftArrow) {
            const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
            setCursorPos(prevSpace === -1 ? 0 : prevSpace + 1);
        }
        if (key.meta && key.rightArrow) {
            const nextSpace = value.indexOf(' ', cursorPos);
            setCursorPos(nextSpace === -1 ? value.length : nextSpace + 1);
        }
        if (key.meta && key.backspace) {
            const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
            const start = prevSpace === -1 ? 0 : prevSpace + 1;
            setValue(value.slice(0, start) + value.slice(cursorPos));
            setCursorPos(start);
        }

        // Interrupt current process or clear line
        if (key.ctrl && input === 'c') {
            if (value.length > 0) {
                setValue('');
                setHistoryIndex(-1);
            } else {
                props.onCancel?.();
            }
        }

        // Emacs / Standard Shortcuts
        if (key.ctrl) {
            if (input === 'k') {
                // Kill to end of line
                const killed = value.slice(cursorPos);
                setValue(value.slice(0, cursorPos));
                if (killed) {
                    setKillRing(prev => [killed, ...prev.slice(0, 9)]);
                    setYankPointer(0);
                }
                return;
            }
            if (input === 'u') {
                // Kill entire line (as per docs)
                const killed = value;
                setValue('');
                setCursorPos(0);
                if (killed) {
                    setKillRing(prev => [killed, ...prev.slice(0, 9)]);
                    setYankPointer(0);
                }
                return;
            }
            if (input === 'y') {
                // Yank
                if (killRing.length > 0) {
                    const toPaste = killRing[yankPointer] || '';
                    const newValue = value.slice(0, cursorPos) + toPaste + value.slice(cursorPos);
                    setValue(newValue);
                    setCursorPos(cursorPos + toPaste.length);
                }
                return;
            }
            if (input === 'g') {
                // Cancel / Clear
                setValue('');
                setCursorPos(0);
                return;
            }
        }

        if (key.meta) {
            if (input === 'y') {
                // Cycle yank (pop/rotate)
                // This is complex to implement perfectly without "last command was yank" state
                // For now, let's just rotate the pointer if we want
                if (killRing.length > 1) {
                    // Undo last yank? That implies we track it. 
                    // Simplified: Just rotate pointer for next yank
                    setYankPointer(prev => (prev + 1) % killRing.length);
                }
            }
            if (input === 'b' || key.leftArrow) {
                const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
                setCursorPos(prevSpace === -1 ? 0 : prevSpace + 1);
                return;
            }
            if (input === 'f' || key.rightArrow) {
                const nextSpace = value.indexOf(' ', cursorPos);
                setCursorPos(nextSpace === -1 ? value.length : nextSpace + 1);
                return;
            }
            if (key.backspace) {
                const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
                const start = prevSpace === -1 ? 0 : prevSpace + 1;
                setValue(value.slice(0, start) + value.slice(cursorPos));
                setCursorPos(start);
                return;
            }
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor={planMode ? theme.planMode : theme.claudeBlue_FOR_SYSTEM_SPINNER} paddingX={1}>
            <Box>
                <Box marginRight={1}>
                    {props.vimModeEnabled && vimMode === 'NORMAL' ? (
                        <Text color={theme.success} bold>[N]</Text>
                    ) : props.vimModeEnabled && vimMode === 'COMMAND' ? (
                        <Text color={theme.warning} bold>[:]</Text>
                    ) : planMode ? (
                        <Text color={theme.planMode}>◈</Text>
                    ) : (
                        <Text color={theme.claudeBlue_FOR_SYSTEM_SPINNER}>❯</Text>
                    )}
                </Box>
                <Box flexDirection="column">
                    {vimMode === 'COMMAND' ? (
                        <Box>
                            <Text color={theme.warning}>:</Text>
                            <TextInput
                                value={commandValue}
                                onChange={setCommandValue}
                                onSubmit={(val) => {
                                    if (val === 'q') props.onCancel?.();
                                    if (val === 'w') { /* save mock maybe? */ }
                                    setVimMode('NORMAL');
                                }}
                            />
                        </Box>
                    ) : (
                        <TextInput
                            focus={!props.vimModeEnabled || vimMode === 'INSERT'}
                            value={value}
                            onChange={(val) => {
                                if (val !== value) {
                                    setUndoStack(prev => [...prev.slice(-49), value]);
                                }
                                setValue(val);
                                if (val !== (history[history.length - 1 - historyIndex] || '')) {
                                    setHistoryIndex(-1);
                                }
                                setCursorPos(val.length);
                            }}
                            onSubmit={(val) => {
                                if (!isSearching) {
                                    setValue('');
                                    setHistoryIndex(-1);
                                    setCursorPos(0);
                                    onSubmit(val);
                                }
                            }}
                        />
                    )}
                    {suggestion && !isSearching && vimMode !== 'COMMAND' && (
                        <Box>
                            <Text dimColor>{" ".repeat(value.length)}{suggestion}</Text>
                        </Box>
                    )}
                </Box>
            </Box>
            {isSearching && (
                <Box paddingLeft={2}>
                    <Text color={theme.warning}>(reverse-i-search): </Text>
                    <Text>{searchQuery}</Text>
                    {searchMatchIndex !== -1 && (
                        <Text dimColor>  [Match: {history[history.length - 1 - searchMatchIndex]}]</Text>
                    )}
                </Box>
            )}
            {value.includes('\n') && (
                <Box paddingLeft={2} marginTop={0}>
                    <Text dimColor italic>Multiline: {value.split('\n').length} lines</Text>
                </Box>
            )}
        </Box>
    );
};
