/**
 * File: src/hooks/useTerminalInput.ts
 * Role: Manages terminal-style input handling (line editing, shortcuts, and exit conditions).
 */

import { useCallback } from 'react';
import { useNotifications } from '../services/terminal/NotificationService.js';

const DOUBLE_ESCAPE_TIMEOUT = 1000;

export interface UseTerminalInputProps {
    value: string;
    onChange: (newValue: string) => void;
    onSubmit?: (value: string) => void;
    onExit?: () => void;
    onClearInput?: () => void;
    onHistoryUp?: () => void;
    onHistoryDown?: () => void;
    multiline?: boolean;
    cursorOffset: number;
    setCursorOffset: (offset: number) => void;
}

export interface UseTerminalInputResult {
    onInput: (char: string, key: any) => void;
}

/**
 * Hook to manage complex terminal input behaviors.
 */
export function useTerminalInput({
    value,
    onChange,
    onSubmit,
    onExit,
    onClearInput,
    onHistoryUp,
    onHistoryDown,
    multiline = false,
    cursorOffset,
    setCursorOffset
}: UseTerminalInputProps): UseTerminalInputResult {
    const { addNotification } = useNotifications();

    const handleCtrlC = useCallback(() => {
        if (value) {
            onChange("");
            setCursorOffset(0);
        } else {
            onExit?.();
        }
    }, [value, onChange, setCursorOffset, onExit]);

    const handleCtrlD = useCallback(() => {
        if (!value) {
            onExit?.();
        }
    }, [value, onExit]);

    const handleDoubleEscape = useCallback(() => {
        addNotification({
            key: "escape-again-to-clear",
            text: "Press Esc again to clear input",
            timeoutMs: DOUBLE_ESCAPE_TIMEOUT
        } as any);
        // Logic for tracking second press would go here
    }, [addNotification]);

    /**
     * Main input handler for the terminal.
     */
    const handleInput = useCallback((char: string, key: any) => {
        if (key.ctrl) {
            switch (char) {
                case 'c': handleCtrlC(); break;
                case 'd': handleCtrlD(); break;
                case 'l': onClearInput?.(); break;
                case 'u': onChange(""); setCursorOffset(0); break;
            }
            return;
        }

        if (key.return) {
            if (multiline && value.endsWith('\\')) {
                onChange(value.slice(0, -1) + '\n');
            } else {
                onSubmit?.(value);
            }
            return;
        }

        if (key.upArrow) onHistoryUp?.();
        if (key.downArrow) onHistoryDown?.();

        if (key.escape) {
            handleDoubleEscape();
        }

        // Basic character insertion (Simplified)
        if (!key.ctrl && !key.meta && char) {
            const newValue = value.slice(0, cursorOffset) + char + value.slice(cursorOffset);
            onChange(newValue);
            setCursorOffset(cursorOffset + char.length);
        }
    }, [value, cursorOffset, onChange, setCursorOffset, handleCtrlC, handleCtrlD, onClearInput, onSubmit, onHistoryUp, onHistoryDown, multiline, handleDoubleEscape]);

    return {
        onInput: handleInput
    };
}
