import { useState, useCallback, useRef, useEffect } from 'react';
import { setCursorStyle, resetCursorStyle } from '../utils/terminal/cursorStyle.js';

export type VimMode = 'INSERT' | 'NORMAL' | 'VISUAL';

export interface UseVimModeProps {
    enabled?: boolean;
    onModeChange?: (mode: VimMode) => void;
    value: string;
    onChange: (value: string) => void;
    cursorOffset: number;
    setCursorOffset: (offset: number) => void;
}

export function useVimMode({ enabled, onModeChange, value, onChange, cursorOffset, setCursorOffset }: UseVimModeProps) {
    const [mode, setMode] = useState<VimMode>('INSERT');
    const commandRef = useRef<string>('');

    // Update hardware cursor style
    useEffect(() => {
        if (!enabled) {
            resetCursorStyle();
            return;
        }

        switch (mode) {
            case 'INSERT':
                setCursorStyle('beam');
                break;
            case 'NORMAL':
            case 'VISUAL':
                setCursorStyle('block');
                break;
        }

        // Cleanup on unmount or disable
        return () => {
            resetCursorStyle();
        };
    }, [enabled, mode]);

    const handleNormalModeInput = useCallback((key: any, input: string) => {
        // Basic Navigation
        if (input === 'h' || key.leftArrow) setCursorOffset(Math.max(0, cursorOffset - 1));
        if (input === 'l' || key.rightArrow) setCursorOffset(Math.min(value.length, cursorOffset + 1));
        if (input === '0') setCursorOffset(0);
        if (input === '$') setCursorOffset(value.length);

        // Mode Switching
        if (input === 'i') {
            setMode('INSERT');
            onModeChange?.('INSERT');
        }
        if (input === 'a') {
            setMode('INSERT');
            onModeChange?.('INSERT');
            setCursorOffset(Math.min(value.length, cursorOffset + 1));
        }

        // Simple Delete (x)
        if (input === 'x') {
            const newValue = value.slice(0, cursorOffset) + value.slice(cursorOffset + 1);
            onChange(newValue);
        }

        // Operators - naive implementation for now
        // DD (delete line)
        if (input === 'd') {
            if (commandRef.current === 'd') {
                onChange('');
                setCursorOffset(0);
                commandRef.current = '';
            } else {
                commandRef.current = 'd';
            }
        } else {
            // Reset command buffer if not a continuation
            if (input !== 'd' && commandRef.current === 'd') {
                commandRef.current = '';
            }
        }
    }, [value, cursorOffset, setCursorOffset, onChange, onModeChange]);

    const handleKey = useCallback((input: string, key: any) => {
        if (!enabled) return false;

        // Escape always goes to Normal mode
        if (key.escape) {
            setMode('NORMAL');
            onModeChange?.('NORMAL');
            // Move cursor back one char if possible when entering normal mode
            setCursorOffset(Math.max(0, cursorOffset - 1));
            return true;
        }

        if (mode === 'NORMAL') {
            handleNormalModeInput(key, input);
            return true; // Suppress default behavior
        }

        return false; // Allow default INSERT behavior
    }, [enabled, mode, onModeChange, cursorOffset, setCursorOffset, handleNormalModeInput]);

    return {
        mode,
        setMode,
        handleKey
    };
}
