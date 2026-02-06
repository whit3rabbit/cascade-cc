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
import { useTerminalInput } from '../../hooks/useTerminalInput.js';

export interface UserPromptMessageProps {
    onSubmit: (value: string) => void;
    onClear?: () => void;
    history?: string[];
    vimModeEnabled?: boolean;
    onVimModeChange?: (mode: 'NORMAL' | 'INSERT') => void;
    planMode?: boolean;
    suggestions?: string[];
    onCancel?: () => void;
    onClearScreen?: () => void;
}

export const UserPromptMessage: React.FC<UserPromptMessageProps> = (props) => {
    const { onSubmit, onClear: _onClear, history = [], planMode = false } = props;
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
    const [lastChar, setLastChar] = useState('');

    // Visual Mode
    const [selectionStart, setSelectionStart] = useState<number | null>(null);

    // Macro Recording
    const [isRecording, setIsRecording] = useState(false);
    const [recordingRegister, setRecordingRegister] = useState<string | null>(null);
    const [recordingPending, setRecordingPending] = useState(false);
    const [macroBuffer, setMacroBuffer] = useState<{ input: string, key: any }[]>([]);
    const [recordedMacros, setRecordedMacros] = useState<Record<string, { input: string, key: any }[]>>({});
    const [playingMacro, setPlayingMacro] = useState(false);
    const [playingMacroPending, setPlayingMacroPending] = useState(false);

    // Buffer Search
    const [isBufferSearching, setIsBufferSearching] = useState(false);
    const [bufferSearchQuery, setBufferSearchQuery] = useState('');
    const [bufferSearchDirection, setBufferSearchDirection] = useState<'forward' | 'backward'>('forward');
    const [lastSearchQuery, setLastSearchQuery] = useState('');

    // ... existing kill ring ...
    const [killRing, setKillRing] = useState<string[]>([]);
    const [yankPointer, setYankPointer] = useState(0);

    // ... existing suggestions ...
    const fuseSuggestions = useMemo(() => new Fuse(props.suggestions || [], { threshold: 0.4 }), [props.suggestions]);
    const fuseHistory = useMemo(() => new Fuse(history, { threshold: 0.4 }), [history]);

    // ... existing effect ...
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

    const playMacro = (register: string) => {
        const macro = recordedMacros[register];
        if (macro) {
            setPlayingMacro(true);
            // In a real terminal, keys are processed sequentially.
            // Here, we'll try to process them. 
            // Note: This might have issues with state batching if not careful.
            for (const m of macro) {
                handleKey(m.input, m.key, true);
            }
            setPlayingMacro(false);
        }
    };

    const handleKey = (input: string, key: any, bypassRecording = false) => {
        // Macro Recording
        if (isRecording && !playingMacro && !bypassRecording) {
            setMacroBuffer(prev => [...prev, { input, key }]);
        }

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
                spawnSync(editor, [tempFile], { stdio: 'inherit' });
                const newValue = readFileSync(tempFile, 'utf8');
                setValue(newValue);
                setCursorPos(newValue.length);
                unlinkSync(tempFile);
            } catch { }
            return;
        }

        // Buffer Search Input Mode
        if (isBufferSearching) {
            if (key.return) {
                setIsBufferSearching(false);
                setLastSearchQuery(bufferSearchQuery);
                performBufferSearch(bufferSearchQuery, bufferSearchDirection);
                return;
            }
            if (key.escape) {
                setIsBufferSearching(false);
                return;
            }
            if (key.backspace) {
                setBufferSearchQuery(prev => prev.slice(0, -1));
                return;
            }
            if (!key.ctrl && !key.meta && input) {
                setBufferSearchQuery(prev => prev + input);
                return;
            }
            return;
        }

        // Macro Register Selection
        if (recordingPending) {
            if (input && !key.ctrl && !key.meta) {
                setRecordingRegister(input);
                setRecordingPending(false);
                setIsRecording(true);
                setMacroBuffer([]);
            }
            return;
        }

        // Macro Playback Selection
        if (playingMacroPending) {
            if (input && !key.ctrl && !key.meta) {
                setPlayingMacroPending(false);
                playMacro(input);
            }
            return;
        }

        // Vim Visual Mode Handling
        if (props.vimModeEnabled && vimMode === 'VISUAL') {
            if (key.escape) {
                setVimMode('NORMAL');
                setSelectionStart(null);
                return;
            }
            if (input === 'h') setCursorPos(Math.max(0, cursorPos - 1));
            if (input === 'l') setCursorPos(Math.min(value.length, cursorPos + 1));
            if (input === 'w') {
                const nextSpace = value.indexOf(' ', cursorPos);
                setCursorPos(nextSpace === -1 ? value.length : nextSpace + 1);
            }
            if (input === 'b') {
                const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
                setCursorPos(prevSpace === -1 ? 0 : prevSpace + 1);
            }
            if (input === '0' || input === '^') setCursorPos(0);
            if (input === '$') setCursorPos(value.length);

            if (input === 'y') {
                const start = Math.min(selectionStart!, cursorPos);
                const end = Math.max(selectionStart!, cursorPos) + 1;
                const selected = value.slice(start, end);
                setKillRing(prev => [selected, ...prev.slice(0, 9)]);
                setVimMode('NORMAL');
                setSelectionStart(null);
            }
            if (input === 'd' || input === 'x') {
                const start = Math.min(selectionStart!, cursorPos);
                const end = Math.max(selectionStart!, cursorPos) + 1;
                setUndoStack(prev => [...prev.slice(-49), value]);
                const newValue = value.slice(0, start) + value.slice(end);
                setValue(newValue);
                setCursorPos(start);
                setVimMode('NORMAL');
                setSelectionStart(null);
            }
            if (input === 'c') {
                const start = Math.min(selectionStart!, cursorPos);
                const end = Math.max(selectionStart!, cursorPos) + 1;
                setUndoStack(prev => [...prev.slice(-49), value]);
                const newValue = value.slice(0, start) + value.slice(end);
                setValue(newValue);
                setCursorPos(start);
                setVimMode('INSERT');
                setSelectionStart(null);
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
            if (input === 'v') {
                setVimMode('VISUAL');
                setSelectionStart(cursorPos);
                return;
            }
            if (input === 'q') {
                if (isRecording) {
                    setIsRecording(false);
                    const finalBuffer = macroBuffer.slice(0, -1); // remove the q if needed, but handleKey is called after
                    setRecordedMacros(prev => ({
                        ...prev,
                        [recordingRegister!]: finalBuffer
                    }));
                    setRecordingRegister(null);
                } else {
                    setRecordingPending(true);
                }
                return;
            }
            if (input === '@') {
                // Play macro pending
                return;
            }
            if (input === '@') {
                setPlayingMacroPending(true);
                return;
            }
            if (input === '/') {
                setIsBufferSearching(true);
                setBufferSearchQuery('');
                setBufferSearchDirection('forward');
                return;
            }
            if (input === '?') {
                setIsBufferSearching(true);
                setBufferSearchQuery('');
                setBufferSearchDirection('backward');
                return;
            }
            if (input === 'n') {
                performBufferSearch(lastSearchQuery, bufferSearchDirection);
                return;
            }
            if (input === 'N') {
                performBufferSearch(lastSearchQuery, bufferSearchDirection === 'forward' ? 'backward' : 'forward');
                return;
            }

            if (input === 'g') {
                if (lastChar === 'g') {
                    setCursorPos(0);
                    setLastChar('');
                } else {
                    setLastChar('g');
                }
                return;
            }
            setLastChar('');

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
            if (input === 'p') {
                if (killRing.length > 0) {
                    const toPaste = killRing[0];
                    const newValue = value.slice(0, cursorPos + 1) + toPaste + value.slice(cursorPos + 1);
                    setValue(newValue);
                    setCursorPos(cursorPos + toPaste.length);
                }
                return;
            }
            if (input === ':') {
                setVimMode('COMMAND');
                setCommandValue('');
                return;
            }
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
            if (input === 'G') {
                setCursorPos(value.length);
                return;
            }

            // Editing
            if (input === 'd') {
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
                const nextIdx = history.slice().reverse().slice(searchMatchIndex + 1).findIndex(h => h.includes(searchQuery));
                if (nextIdx !== -1) setSearchMatchIndex(searchMatchIndex + 1 + nextIdx);
            }
            return;
        }

        // Search Mode
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

        // macOS style word navigation
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

        if (key.ctrl && input === 'c') {
            if (value.length > 0) {
                setValue('');
                setHistoryIndex(-1);
            } else {
                props.onCancel?.();
            }
        }

        // Emacs Shortcuts
        if (key.ctrl) {
            if (input === 'a') {
                setCursorPos(0);
                return;
            }
            if (input === 'e') {
                setCursorPos(value.length);
                return;
            }
            if (input === 'b') {
                setCursorPos(Math.max(0, cursorPos - 1));
                return;
            }
            if (input === 'f') {
                setCursorPos(Math.min(value.length, cursorPos + 1));
                return;
            }
            if (input === 'd') {
                if (value.length === 0) {
                    props.onCancel?.();
                } else if (cursorPos < value.length) {
                    const newValue = value.slice(0, cursorPos) + value.slice(cursorPos + 1);
                    setValue(newValue);
                }
                return;
            }
            if (input === 'l') {
                props.onClearScreen?.();
                return;
            }
            if (input === 'p') {
                if (historyIndex < history.length - 1) {
                    const newIndex = historyIndex + 1;
                    setHistoryIndex(newIndex);
                    setValue(history[history.length - 1 - newIndex] || '');
                }
                return;
            }
            if (input === 'n') {
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
            if (input === 'w') {
                const prevSpace = value.lastIndexOf(' ', cursorPos - 2);
                const start = prevSpace === -1 ? 0 : prevSpace + 1;
                setValue(value.slice(0, start) + value.slice(cursorPos));
                setCursorPos(start);
                return;
            }
            if (input === 'k') {
                const killed = value.slice(cursorPos);
                setValue(value.slice(0, cursorPos));
                if (killed) {
                    setKillRing(prev => [killed, ...prev.slice(0, 9)]);
                    setYankPointer(0);
                }
                return;
            }
            if (input === 'u') {
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
                if (killRing.length > 0) {
                    const toPaste = killRing[yankPointer] || '';
                    const newValue = value.slice(0, cursorPos) + toPaste + value.slice(cursorPos);
                    setValue(newValue);
                    setCursorPos(cursorPos + toPaste.length);
                }
                return;
            }
        }

        if (key.meta) {
            if (input === 'y') {
                if (killRing.length > 1) {
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
        }
        if (key.leftArrow && !key.meta) {
            setCursorPos(Math.max(0, cursorPos - 1));
            return;
        }
        if (key.rightArrow && !key.meta) {
            setCursorPos(Math.min(value.length, cursorPos + 1));
            return;
        }

        if (key.backspace) {
            if (cursorPos > 0) {
                const newValue = value.slice(0, cursorPos - 1) + value.slice(cursorPos);
                setValue(newValue);
                setCursorPos(cursorPos - 1);
            }
            return;
        }

        if (key.delete) {
            if (cursorPos < value.length) {
                const newValue = value.slice(0, cursorPos) + value.slice(cursorPos + 1);
                setValue(newValue);
            }
            return;
        }

        if (key.return && !key.shift && !key.meta) {
            onSubmit(value);
            setValue('');
            setCursorPos(0);
            setHistoryIndex(-1);
            return;
        }

        // Standard character input
        if (input && !key.ctrl && !key.meta && !key.escape && !key.tab && !key.return) {
            if (vimMode === 'INSERT' || !props.vimModeEnabled) {
                const newValue = value.slice(0, cursorPos) + input + value.slice(cursorPos);
                setValue(newValue);
                setCursorPos(cursorPos + input.length);
                return;
            }
        }
    };

    const performBufferSearch = (query: string, direction: 'forward' | 'backward') => {
        if (!query) return;
        let searchIndex = -1;
        if (direction === 'forward') {
            searchIndex = value.indexOf(query, cursorPos + 1);
            if (searchIndex === -1) searchIndex = value.indexOf(query); // wrap
        } else {
            searchIndex = value.lastIndexOf(query, cursorPos - 1);
            if (searchIndex === -1) searchIndex = value.lastIndexOf(query); // wrap
        }
        if (searchIndex !== -1) setCursorPos(searchIndex);
    };

    const { onInput: terminalInputHandler } = useTerminalInput({
        value,
        onChange: setValue,
        onSubmit: (val: string) => {
            onSubmit(val);
            setValue('');
            setCursorPos(0);
            setHistoryIndex(-1);
        },
        onExit: props.onCancel || (() => { }),
        onClearInput: () => setValue(''),
        onHistoryUp: () => {
            if (historyIndex < history.length - 1) {
                const newIndex = historyIndex + 1;
                setHistoryIndex(newIndex);
                setValue(history[history.length - 1 - newIndex] || '');
                setCursorPos(0);
            }
        },
        onHistoryDown: () => {
            if (historyIndex > -1) {
                const newIndex = historyIndex - 1;
                setHistoryIndex(newIndex);
                if (newIndex === -1) {
                    setValue('');
                } else {
                    setValue(history[history.length - 1 - newIndex] || '');
                }
                setCursorPos(0);
            }
        },
        multiline: true,
        cursorOffset: cursorPos,
        setCursorOffset: setCursorPos
    });

    useInput((input, key) => {
        // Handle search and vim modes first
        if (isSearching || isBufferSearching || recordingPending || playingMacroPending || (props.vimModeEnabled && vimMode !== 'INSERT')) {
            handleKey(input, key);
            return;
        }

        // Use unified terminal input handler for most actions
        terminalInputHandler(input, key);
    });

    const renderValue = () => {
        if (vimMode === 'VISUAL' && selectionStart !== null) {
            const start = Math.min(selectionStart, cursorPos);
            const end = Math.max(selectionStart, cursorPos) + 1;
            return (
                <Text>
                    {value.slice(0, start)}
                    <Text backgroundColor="white" color="black">{value.slice(start, end)}</Text>
                    {value.slice(end)}
                </Text>
            );
        }

        // Show cursor in Normal Mode
        if (props.vimModeEnabled && vimMode === 'NORMAL') {
            return (
                <Text>
                    {value.slice(0, cursorPos)}
                    <Text backgroundColor="white" color="black">
                        {value[cursorPos] || ' '}
                    </Text>
                    {value.slice(cursorPos + 1)}
                </Text>
            );
        }

        return <Text>{value}</Text>;
    };

    return (
        <Box flexDirection="column" borderStyle="round" borderColor={planMode ? theme.planMode : theme.claudeBlue_FOR_SYSTEM_SPINNER} paddingX={1}>
            <Box>
                <Box marginRight={1}>
                    {isRecording && <Text color="red">● </Text>}
                    {props.vimModeEnabled && vimMode === 'NORMAL' ? (
                        <Text color={theme.success} bold>[N]</Text>
                    ) : props.vimModeEnabled && vimMode === 'COMMAND' ? (
                        <Text color={theme.warning} bold>[:]</Text>
                    ) : props.vimModeEnabled && vimMode === 'VISUAL' ? (
                        <Text color="cyan" bold>[V]</Text>
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
                                    const cmd = val.trim();
                                    if (cmd === 'q' || cmd === 'q!' || cmd === 'qa') {
                                        props.onCancel?.();
                                    } else if (cmd === 'w') {
                                        // In a REPL, :w usually implies "submit" or just "save buffer" (no-op here since we auto-save history)
                                        // We'll treat it as submit for now or just return to normal
                                        setVimMode('NORMAL');
                                    } else if (cmd === 'wq' || cmd === 'x') {
                                        props.onSubmit(value); // Submit as "save"
                                        // props.onCancel?.(); // Then quit? REPL semantics are tricky. 
                                        // Actually :wq usually means write file and quit editor. 
                                        // Here it might mean "submit query".
                                    } else {
                                        // Unknown command
                                    }
                                    setVimMode('NORMAL');
                                }}
                            />
                        </Box>
                    ) : isBufferSearching ? (
                        <Box>
                            <Text color="yellow">{bufferSearchDirection === 'forward' ? '/' : '?'}</Text>
                            <Text>{bufferSearchQuery}</Text>
                        </Box>
                    ) : (
                        <Box>
                            {renderValue()}
                        </Box>
                    )}
                    {suggestion && !isSearching && vimMode !== 'COMMAND' && !isBufferSearching && (
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
            {recordingPending && (
                <Box paddingLeft={2}>
                    <Text color="red">Recording into register...</Text>
                </Box>
            )}
            {playingMacroPending && (
                <Box paddingLeft={2}>
                    <Text color="cyan">Play macro from register...</Text>
                </Box>
            )}
        </Box>
    );
};
