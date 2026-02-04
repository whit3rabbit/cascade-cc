import { useState, useCallback, useRef } from 'react';

export interface UseArrowKeyHistoryOptions {
    history: string[];
    currentValue: string;
    onHistoryChange: (value: string) => void;
    onCursorOffsetChange: (offset: number) => void;
}

/**
 * Hook to manage arrow-key history navigation, mirroring IfK from golden source.
 */
export function useArrowKeyHistory({
    history,
    currentValue,
    onHistoryChange,
    onCursorOffsetChange
}: UseArrowKeyHistoryOptions) {
    const [index, setIndex] = useState(0);
    const stash = useRef<string | undefined>(undefined);

    const onHistoryUp = useCallback(() => {
        if (index === 0) {
            stash.current = currentValue;
        }

        if (index < history.length) {
            const nextIndex = index + 1;
            const historyValue = history[history.length - nextIndex];
            if (historyValue !== undefined) {
                setIndex(nextIndex);
                onHistoryChange(historyValue);
                onCursorOffsetChange(historyValue.length);
            }
        }
    }, [index, history, currentValue, onHistoryChange, onCursorOffsetChange]);

    const onHistoryDown = useCallback(() => {
        if (index === 0) return;

        const nextIndex = index - 1;
        setIndex(nextIndex);

        if (nextIndex === 0) {
            const val = stash.current || "";
            onHistoryChange(val);
            onCursorOffsetChange(val.length);
            stash.current = undefined;
        } else {
            const historyValue = history[history.length - nextIndex];
            if (historyValue !== undefined) {
                onHistoryChange(historyValue);
                onCursorOffsetChange(historyValue.length);
            }
        }
    }, [index, history, onHistoryChange, onCursorOffsetChange]);

    const resetHistory = useCallback(() => {
        setIndex(0);
        stash.current = undefined;
    }, []);

    return {
        historyIndex: index,
        onHistoryUp,
        onHistoryDown,
        resetHistory
    };
}
