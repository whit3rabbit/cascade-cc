import { useState, useCallback, useRef } from 'react';

export interface UndoRedoState {
    text: string;
    cursorOffset: number;
    timestamp: number;
}

export interface UseUndoRedoOptions {
    maxBufferSize?: number;
    debounceMs?: number;
}

/**
 * Hook to manage undo/redo buffer for text input, mirroring mNK from golden source.
 */
export function useUndoRedo({
    maxBufferSize = 100,
    debounceMs = 200
}: UseUndoRedoOptions = {}) {
    const [history, setHistory] = useState<UndoRedoState[]>([]);
    const [index, setIndex] = useState(-1);
    const lastPushTime = useRef(0);
    const timer = useRef<NodeJS.Timeout | null>(null);

    const pushToBuffer = useCallback((text: string, cursorOffset: number) => {
        const now = Date.now();
        if (timer.current) {
            clearTimeout(timer.current);
            timer.current = null;
        }

        if (now - lastPushTime.current < debounceMs) {
            timer.current = setTimeout(() => {
                pushToBuffer(text, cursorOffset);
            }, debounceMs);
            return;
        }

        lastPushTime.current = now;
        setHistory(prev => {
            const truncated = index >= 0 ? prev.slice(0, index + 1) : prev;
            const last = truncated[truncated.length - 1];

            if (last && last.text === text) {
                return truncated;
            }

            const next = [...truncated, { text, cursorOffset, timestamp: now }];
            if (next.length > maxBufferSize) {
                return next.slice(-maxBufferSize);
            }
            return next;
        });

        setIndex(prev => {
            const next = prev >= 0 ? prev + 1 : history.length;
            return Math.min(next, maxBufferSize - 1);
        });
    }, [debounceMs, maxBufferSize, index, history.length]);

    const undo = useCallback((): UndoRedoState | undefined => {
        if (index < 0 || history.length === 0) return undefined;

        const nextIndex = Math.max(0, index - 1);
        const state = history[nextIndex];
        if (state) {
            setIndex(nextIndex);
            return state;
        }
        return undefined;
    }, [history, index]);

    const redo = useCallback((): UndoRedoState | undefined => {
        if (index >= history.length - 1) return undefined;

        const nextIndex = index + 1;
        const state = history[nextIndex];
        if (state) {
            setIndex(nextIndex);
            return state;
        }
        return undefined;
    }, [history, index]);

    const clearBuffer = useCallback(() => {
        setHistory([]);
        setIndex(-1);
        lastPushTime.current = 0;
        if (timer.current) {
            clearTimeout(timer.current);
            timer.current = null;
        }
    }, []);

    return {
        pushToBuffer,
        undo,
        redo,
        canUndo: index > 0 && history.length > 1,
        canRedo: index < history.length - 1,
        clearBuffer
    };
}
