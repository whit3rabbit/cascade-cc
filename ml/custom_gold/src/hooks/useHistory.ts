
import { useState, useCallback, useRef } from 'react';
import { getProjectHistory } from '../services/terminal/HistoryService.js';

/**
 * Hook to manage history navigation state.
 */
export function useHistory(projectPath: string) {
    const [history, setHistory] = useState<string[]>([]);
    const [index, setIndex] = useState(-1);
    const [tempValue, setTempValue] = useState("");
    const isLoaded = useRef(false);

    const load = useCallback(async () => {
        if (isLoaded.current) return;
        try {
            const entries = await getProjectHistory(projectPath);
            setHistory(entries.map(e => e.display));
            isLoaded.current = true;
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }, [projectPath]);

    const up = useCallback((currentValue: string): string | null => {
        if (!isLoaded.current) {
            load();
            return null;
        }

        let newIndex = index;
        if (index === -1) {
            setTempValue(currentValue);
            newIndex = 0;
        } else {
            newIndex = Math.min(index + 1, history.length - 1);
        }

        if (newIndex < history.length) {
            setIndex(newIndex);
            return history[newIndex];
        }
        return null;
    }, [history, index, load]);

    const down = useCallback((): string | null => {
        if (index === -1) return null;

        const nextIndex = index - 1;
        setIndex(nextIndex);

        if (nextIndex === -1) {
            return tempValue;
        }
        return history[nextIndex];
    }, [history, index, tempValue]);

    const reset = useCallback(() => {
        setIndex(-1);
        setTempValue("");
    }, []);

    return { up, down, reset, load };
}
