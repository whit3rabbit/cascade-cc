/**
 * File: src/hooks/useHistory.tsx
 * Role: Hook for managing terminal command history navigation and search.
 */

import { useState, useCallback, useEffect } from 'react';
import { useInput } from 'ink';
import { getProjectHistory } from '../services/terminal/HistoryService.js';
import { HistoryEntry } from '../services/terminal/HistoryService.js';

export interface UseHistoryProps {
    projectPath: string;
    onSelect: (value: string) => void;
}

export interface UseHistoryResult {
    isNavigatingHistory: boolean;
    resetHistoryNavigation: () => void;
}

/**
 * Hook to manage history navigation (Up/Down) in the terminal.
 */
export function useHistory({ projectPath, onSelect }: UseHistoryProps): UseHistoryResult {
    const [history, setHistory] = useState<HistoryEntry[]>([]);
    const [historyIndex, setHistoryIndex] = useState<number>(-1);
    const [storedInput, setStoredInput] = useState<string>("");

    // Load history on mount or when project path changes
    useEffect(() => {
        let mounted = true;
        getProjectHistory(projectPath).then(data => {
            if (mounted) setHistory(data);
        });
        return () => { mounted = false; };
    }, [projectPath]);

    /**
     * Resets the history navigation state.
     */
    const resetHistoryNavigation = useCallback(() => {
        setHistoryIndex(-1);
        setStoredInput("");
    }, []);

    // Handle Up/Down keys for history navigation
    useInput((input, key) => {
        if (key.upArrow) {
            setHistoryIndex(prev => {
                if (prev === -1) setStoredInput(input);
                const next = prev + 1;
                if (next < history.length) {
                    onSelect(history[next].display);
                    return next;
                }
                return prev;
            });
        }

        if (key.downArrow) {
            setHistoryIndex(prev => {
                const next = prev - 1;
                if (next >= 0) {
                    onSelect(history[next].display);
                    return next;
                }
                if (next === -1) {
                    onSelect(storedInput);
                    return -1;
                }
                return prev;
            });
        }
    });

    return {
        isNavigatingHistory: historyIndex !== -1,
        resetHistoryNavigation
    };
}

export default useHistory;
