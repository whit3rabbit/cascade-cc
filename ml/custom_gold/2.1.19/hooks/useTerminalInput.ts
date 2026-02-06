import { useCallback, useEffect, useRef } from 'react';
import { useNotifications } from '../services/terminal/NotificationService.js';
import { useDoubleAction } from './useDoubleAction.js';
import { useUndoRedo } from './useUndoRedo.js';
import { useArrowKeyHistory } from './useArrowKeyHistory.js';
import { useSuggestions } from './useSuggestions.js';
import { useMacros } from './useMacros.js';
import { useKillRing } from './useKillRing.js';
import { useVimMode, VimMode } from './useVimMode.js';
import { useStash } from './useStash.js';

export interface TerminalInputProps {
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    onExit?: () => void;
    onClearInput?: () => void;
    onHistoryUp?: () => void;
    onHistoryDown?: () => void;
    multiline?: boolean;
    cursorOffset: number;
    setCursorOffset: (offset: number) => void;
    history?: string[];
    commands?: any[];
    agents?: any[];
    vimModeEnabled?: boolean;
    onVimModeChange?: (mode: VimMode) => void;
}

/**
 * Hook to manage complex terminal input behaviors, including Emacs keybindings,
 * double-press actions (Ctrl-C, Escape), and multiline handling.
 * Aligned with 2.1.19 gold reference (chunk1174).
 */
export function useTerminalInput({
    value,
    onChange,
    onSubmit,
    onExit,
    onClearInput,
    onHistoryUp: _onHistoryUp,
    onHistoryDown,
    multiline = false,
    cursorOffset,
    setCursorOffset,
    history = [],
    commands = [],
    agents = [],
    vimModeEnabled = false,
    onVimModeChange
}: TerminalInputProps) {
    const { addNotification, removeNotification } = useNotifications();
    const { undo, redo } = useUndoRedo();
    const killRing = useKillRing();
    const lastKeyWasYank = useRef(false);

    const {
        onHistoryUp: internalOnHistoryUp,
        onHistoryDown: internalOnHistoryDown
    } = useArrowKeyHistory({
        history,
        currentValue: value,
        onHistoryChange: onChange,
        onCursorOffsetChange: setCursorOffset
    });

    const {
        suggestions,
        selectedSuggestion,
        onNextSuggestion,
        onPrevSuggestion,
        resetSuggestions
    } = useSuggestions({
        input: value,
        commands,
        agents
    });

    const stash = useStash({
        value,
        onChange,
        onNotify: (msg) => {
            addNotification({ key: 'stash', text: msg, timeoutMs: 2000 });
        }
    });

    // Use a ref to break the dependency cycle between handleInput and macros
    const handleInputRef = useRef<((char: string, key: any, bypassRecording?: boolean) => void) | null>(null);
    const onInputProxyRef = useCallback((char: string, key: any, bypassRecording: boolean = false) => {
        if (handleInputRef.current) {
            handleInputRef.current(char, key, bypassRecording);
        }
    }, []);

    const macros = useMacros({ onInput: onInputProxyRef });

    const handleUndo = useCallback(() => {
        const state = undo();
        if (state) {
            onChange(state.text);
            setCursorOffset(state.cursorOffset);
        }
    }, [undo, onChange, setCursorOffset]);

    const vimMode = useVimMode({
        enabled: vimModeEnabled,
        onModeChange: onVimModeChange,
        value,
        onChange,
        cursorOffset,
        setCursorOffset,
        killRing,
        macros,
        onSubmit,
        onExit,
        onUndo: handleUndo
    });

    const onExitConfirm = useCallback(() => {
        onExit?.();
    }, [onExit]);

    const onCtrlCFirstPress = useCallback(() => {
        if (value) {
            onChange("");
            setCursorOffset(0);
        }
    }, [value, onChange, setCursorOffset]);

    // ... (keep DoubleActions) ...

    const handleCtrlC = useDoubleAction(
        (pending) => {
            if (pending) {
                addNotification({
                    key: "exit-confirmation",
                    text: "Press Ctrl-C again to exit",
                    timeoutMs: 800
                });
            } else {
                removeNotification("exit-confirmation");
            }
        },
        onExitConfirm,
        onCtrlCFirstPress,
        800
    );

    const handleEscape = useDoubleAction(
        (pending) => {
            if (pending) {
                addNotification({
                    key: "escape-again-to-clear",
                    text: "Esc to clear again",
                    timeoutMs: 1000
                });
            } else {
                removeNotification("escape-again-to-clear");
            }
        },
        () => {
            // Confirm: double escape to clear input
            onChange("");
            setCursorOffset(0);
            onClearInput?.();
        },
        undefined,
        1000
    );

    /**
     * Main input handler for the terminal.
     */
    const handleInput = useCallback((char: string, key: any, bypassRecording: boolean = false) => {
        // Vim Mode Key Handling
        if (vimModeEnabled && vimMode.handleKey(char, key)) {
            return;
        }

        // Record step if recording and not playing back
        if (!bypassRecording) {
            macros.recordStep(char, key);
        }

        // Reset kill action if not a kill-related key
        if (!key.ctrl || (char !== 'k' && char !== 'u' && char !== 'w')) {
            if (char !== 'y' || !key.ctrl) {
                killRing.resetKillAction();
            }
        }

        // Track yank sequence for Meta-y
        const currentKeyIsYank = key.ctrl && char === 'y';
        if (!currentKeyIsYank) {
            lastKeyWasYank.current = false;
        }

        if (key.ctrl) {
            switch (char) {
                case 's': // Stash (Category 2)
                    stash.handleStash();
                    break;
                case 'c':
                    handleCtrlC();
                    break;
                case 'd':
                    if (!value) onExit?.();
                    break;
                case 'a': // Home (Emacs)
                    setCursorOffset(0);
                    break;
                case 'e': // End (Emacs)
                    setCursorOffset(value.length);
                    break;
                case 'b': // Left (Emacs)
                    setCursorOffset(Math.max(0, cursorOffset - 1));
                    break;
                case 'f': // Right (Emacs)
                    setCursorOffset(Math.min(value.length, cursorOffset + 1));
                    break;
                case 'u': // Delete to line start (Emacs)
                    {
                        const killed = value.slice(0, cursorOffset);
                        killRing.push(killed, 'prepend');
                        onChange(value.slice(cursorOffset));
                        setCursorOffset(0);
                    }
                    break;
                case 'k': // Delete to line end (Emacs)
                    {
                        const killed = value.slice(cursorOffset);
                        killRing.push(killed, 'append');
                        onChange(value.slice(0, cursorOffset));
                    }
                    break;
                case 'w': // Delete word before (Emacs)
                    {
                        const before = value.slice(0, cursorOffset);
                        const match = before.match(/(\s*\S+)\s*$/);
                        if (match) {
                            const gap = match[0].length;
                            killRing.push(match[0], 'prepend');
                            onChange(value.slice(0, cursorOffset - gap) + value.slice(cursorOffset));
                            setCursorOffset(cursorOffset - gap);
                        } else if (before.length > 0) {
                            killRing.push(before, 'prepend');
                            onChange(value.slice(cursorOffset));
                            setCursorOffset(0);
                        }
                    }
                    break;
                case 'l': // Clear screen/input
                    onClearInput?.();
                    break;
                case 'n': // Next history
                    onHistoryDown?.();
                    break;
                case 'p': // Previous history
                    internalOnHistoryUp();
                    break;
                case 'z': // Undo
                    {
                        const state = undo();
                        if (state) {
                            onChange(state.text);
                            setCursorOffset(state.cursorOffset);
                        }
                    }
                    break;
                case '_': // Undo (alternative, as per research)
                    {
                        const state = undo();
                        if (state) {
                            onChange(state.text);
                            setCursorOffset(state.cursorOffset);
                        }
                    }
                    break;
                case 'r': // Redo (Aligned with Gold)
                    {
                        const state = redo();
                        if (state) {
                            onChange(state.text);
                            setCursorOffset(state.cursorOffset);
                        }
                    }
                    break;
                case 'y': // Yank (Emacs)
                    {
                        const text = killRing.yank();
                        if (text) {
                            const newValue = value.slice(0, cursorOffset) + text + value.slice(cursorOffset);
                            onChange(newValue);
                            setCursorOffset(cursorOffset + text.length);
                            lastKeyWasYank.current = true;
                        }
                    }
                    break;
                case 'q': // Toggle Macro Recording
                    if (macros.isRecording) {
                        macros.stopRecording();
                        addNotification({ key: 'macro-stop', text: 'Macro recording stopped', timeoutMs: 1500 });
                    } else {
                        macros.startRecording('q'); // Default to 'q' register
                        addNotification({ key: 'macro-start', text: 'Macro recording started...', timeoutMs: 1500 });
                    }
                    break;
            }
            return;
        }

        // ... (keep existing Meta checks) ...

        if (key.meta) {
            switch (char) {
                case 'b': // Back word (Alt-b)
                    {
                        const before = value.slice(0, cursorOffset);
                        const match = before.match(/(\S+\s*)$/);
                        if (match) {
                            setCursorOffset(cursorOffset - match[0].length);
                        } else {
                            setCursorOffset(0);
                        }
                    }
                    break;
                case 'f': // Forward word (Alt-f)
                    {
                        const after = value.slice(cursorOffset);
                        const match = after.match(/^(\s*\S+)/);
                        if (match) {
                            setCursorOffset(cursorOffset + match[0].length);
                        } else {
                            setCursorOffset(value.length);
                        }
                    }
                    break;
                case 'd': // Delete word after (Alt-d)
                    {
                        const after = value.slice(cursorOffset);
                        const match = after.match(/^(\s*\S+)/);
                        if (match) {
                            onChange(value.slice(0, cursorOffset) + value.slice(cursorOffset + match[0].length));
                        }
                    }
                    break;
                case 'y': // Yank-pop (Emacs)
                    if (lastKeyWasYank.current) {
                        const currentYank = killRing.yank();
                        const nextYank = killRing.rotate();
                        if (currentYank && nextYank) {
                            const start = cursorOffset - currentYank.length;
                            const newValue = value.slice(0, start) + nextYank + value.slice(cursorOffset);
                            onChange(newValue);
                            setCursorOffset(start + nextYank.length);
                        }
                    }
                    break;
            }
            return;
        }

        if (key.return) {
            // Shift+Enter or Option+Enter (Meta+Enter) for newline
            if (key.shift || key.meta) {
                const newValue = value.slice(0, cursorOffset) + '\n' + value.slice(cursorOffset);
                onChange(newValue);
                setCursorOffset(cursorOffset + 1);
                return;
            }

            // Multiline escape logic
            if (value.slice(0, cursorOffset).endsWith('\\')) {
                const newValue = value.slice(0, cursorOffset - 1) + '\n' + value.slice(cursorOffset);
                onChange(newValue);
                setCursorOffset(cursorOffset);
                return;
            }
            onSubmit?.(value);
            return;
        }

        if (char === '@' && !key.ctrl && !key.meta) {
            macros.playMacro('q');
            return;
        }

        if (key.upArrow && !key.shift) {
            if (suggestions.length > 0) {
                onPrevSuggestion();
            } else {
                internalOnHistoryUp();
            }
            return;
        }
        if (key.downArrow && !key.shift) {
            if (suggestions.length > 0) {
                onNextSuggestion();
            } else {
                internalOnHistoryDown();
            }
            return;
        }
        if (key.leftArrow) setCursorOffset(Math.max(0, cursorOffset - 1));
        if (key.rightArrow) setCursorOffset(Math.min(value.length, cursorOffset + 1));

        if (key.backspace) {
            if (cursorOffset > 0) {
                onChange(value.slice(0, cursorOffset - 1) + value.slice(cursorOffset));
                setCursorOffset(cursorOffset - 1);
            }
        }
        if (key.delete) {
            if (cursorOffset < value.length) {
                onChange(value.slice(0, cursorOffset) + value.slice(cursorOffset + 1));
            }
        }

        if (key.escape) {
            // If in Vim mode and not Normal, delegate to Vim handler via early return above.
            // If here, we are either in standard mode or Vim Normal mode (handled above too?)
            // Actually, handleKey returns true if it handled it.
            // But we also have double-escape to clear.
            handleEscape();
        }

        // Basic character insertion (only if not a complex key)
        if (!key.ctrl && !key.meta && char && char.length === 1 && !key.return) {
            const newValue = value.slice(0, cursorOffset) + char + value.slice(cursorOffset);
            onChange(newValue);
            setCursorOffset(cursorOffset + 1);
        }
    }, [
        value,
        cursorOffset,
        onChange,
        setCursorOffset,
        handleCtrlC,
        handleEscape,
        onClearInput,
        onSubmit,
        internalOnHistoryUp,
        internalOnHistoryDown,
        multiline,
        onExit,
        undo,
        redo,
        suggestions,
        onNextSuggestion,
        onPrevSuggestion,
        resetSuggestions,
        macros,
        killRing,
        stash,
        vimModeEnabled,
        vimMode
    ]);

    // Assign handleInput to ref for macro playback to avoid circular dependency
    useEffect(() => {
        handleInputRef.current = handleInput;
    }, [handleInput]);

    useEffect(() => {
        if (!value.startsWith('/')) {
            resetSuggestions();
        }
    }, [value, resetSuggestions]);

    return {
        onInput: handleInput,
        suggestions,
        selectedSuggestion,
        isMacroRecording: macros.isRecording,
        currentVimMode: vimModeEnabled ? vimMode.mode : undefined,
        stashedValue: stash.stashedValue
    };
}
