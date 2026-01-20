// Logic from chunk_498.ts (Autocomplete Hook)

import { useState, useCallback, useRef, useEffect } from 'react';
import { useInput } from 'ink';
import { useNotifications } from '../services/terminal/NotificationService.js';
import { suggestSlashCommands, suggestFiles, suggestDirectories } from '../services/terminal/FileIndexService.js';
// import { fetchSuggestions } from '../services/terminal/BashExecutor.js'; // Placeholder for shell suggestions

const DEBOUNCE_MS = 200;

export function useAutocomplete({
    commands,
    onInputChange,
    onSubmit,
    setCursorOffset,
    input,
    cursorOffset,
    mode,
    agents,
    suggestionsState,
    setSuggestionsState,
    suppressSuggestions = false,
    markAccepted
}: any) {
    const { addNotification } = useNotifications();
    const [suggestionType, setSuggestionType] = useState<string>("none");
    const [maxColumnWidth, setMaxColumnWidth] = useState<number | undefined>(undefined);
    const lastCursorOffset = useRef(cursorOffset);

    // Track debounced logic
    const fetchRef = useRef<string>("");

    // Clear suggestions helper
    const clearSuggestions = useCallback(() => {
        setSuggestionsState({
            commandArgumentHint: undefined,
            suggestions: [],
            selectedSuggestion: -1
        });
        setSuggestionType("none");
        setMaxColumnWidth(undefined);
    }, [setSuggestionsState]);

    // Main suggestion update logic
    const updateSuggestions = useCallback(async (text: string, offset: number, checkDebounce = false) => {
        if (suppressSuggestions) {
            clearSuggestions();
            return;
        }

        fetchRef.current = text;
        const currentRef = text;

        // Helpers for finding completion token
        const upToCursor = text.substring(0, offset);
        const isAtCursor = offset === text.length && offset > 0 && text.length > 0 && text[offset - 1] === ' ';

        // 1. Directory completion for /add-dir
        if (mode === 'prompt' && text.startsWith('/') && offset > 0) {
            // Simplified parsing for add-dir
            if (text.startsWith('/add-dir ')) {
                const arg = text.substring('/add-dir '.length);
                if (arg.match(/\s+$/)) {
                    clearSuggestions();
                    return;
                }
                const dirs = await suggestDirectories(arg);
                if (currentRef !== fetchRef.current) return;

                if (dirs.length > 0) {
                    setSuggestionsState((prev: any) => ({
                        suggestions: dirs,
                        selectedSuggestion: adjustSelection(prev.suggestions, prev.selectedSuggestion, dirs),
                        commandArgumentHint: undefined
                    }));
                    setSuggestionType('directory');
                    return;
                }
                clearSuggestions();
                return;
            }
        }

        // 2. Command Completion
        if (mode === 'prompt' && text.startsWith('/') && offset > 0 && !isAtCursor) {
            const suggestions = await suggestSlashCommands(text, commands);
            if (currentRef !== fetchRef.current) return;

            let hint = undefined;
            // Logic to find exact match hint would go here

            setSuggestionsState((prev: any) => ({
                commandArgumentHint: hint,
                suggestions: suggestions,
                selectedSuggestion: adjustSelection(prev.suggestions, prev.selectedSuggestion, suggestions)
            }));

            setSuggestionType(suggestions.length > 0 ? 'command' : 'none');
            if (suggestions.length > 0) {
                const maxLen = Math.max(...suggestions.map((s: any) => s.displayText.length));
                setMaxColumnWidth(maxLen + 5);
            } else {
                setMaxColumnWidth(undefined);
            }
            return;
        }

        // 3. File Completion (General Argument)
        if (suggestionType === 'command') {
            // Cancel invalid command state?
            clearSuggestions();
        }

        // Trigger file suggestions if we have @ or just generally if needed?
        // Logic from chunk_498 implies logic for @ detection:
        const atMatch = upToCursor.match(/(^|\s)@([a-zA-Z0-9_\-./\\()[\]~]*|"[^"]*"?)$/);

        if (atMatch || (suggestionType === 'file')) {
            // simplified extraction
            const query = atMatch ? atMatch[2] : (text.split(' ').pop() || "");
            const files = await suggestFiles(query);
            // TODO: filter out agents, mcp hints here similar to P$0

            if (currentRef !== fetchRef.current) return;

            if (files.length === 0) {
                clearSuggestions();
                return;
            }

            setSuggestionsState((prev: any) => ({
                suggestions: files,
                // preserve selection if possible
                selectedSuggestion: adjustSelection(prev.suggestions, prev.selectedSuggestion, files),
                commandArgumentHint: undefined
            }));
            setSuggestionType('file');
            setMaxColumnWidth(undefined);
            return;
        }

        // 4. Shell Completion
        if (mode === 'bash' || suggestionType === 'shell') {
            // call shell completion stub
            // const shellSuggestions = await fetchSuggestions(text, ...);
            // ... logic
            clearSuggestions(); // Stub
            return;
        }

        // Default: clear
        clearSuggestions();
    }, [mode, commands, suppressSuggestions, suggestionType, clearSuggestions, setSuggestionsState]);

    // Use debounce for updating
    useEffect(() => {
        const timer = setTimeout(() => {
            updateSuggestions(input, cursorOffset, true);
        }, DEBOUNCE_MS);
        return () => clearTimeout(timer);
    }, [input, cursorOffset, updateSuggestions]);

    // Selection Handling
    const handleSelection = useCallback(() => {
        const { suggestions, selectedSuggestion } = suggestionsState;
        if (selectedSuggestion < 0 || suggestions.length === 0) return;

        const item = suggestions[selectedSuggestion];

        // Apply completion logic based on suggestionType
        if (suggestionType === 'command') {
            // Replace text with command
            const toInsert = item.displayText + " ";
            onInputChange(toInsert);
            setCursorOffset(toInsert.length);
            clearSuggestions();
        } else if (suggestionType === 'directory' || suggestionType === 'file') {
            // Fill path
            // Logic to find replacement range... simplified:
            const parts = input.split(' ');
            parts.pop(); // remove partial
            parts.push(item.displayText);
            const newText = parts.join(' '); // very naive
            onInputChange(newText);
            setCursorOffset(newText.length);
            // Trigger next level suggestions
            updateSuggestions(newText, newText.length);
            // Don't fully clear if we want continuous completion?
        }

    }, [suggestionsState, suggestionType, input, onInputChange, setCursorOffset, clearSuggestions, updateSuggestions]);

    useInput((input, key) => {
        const { suggestions, selectedSuggestion } = suggestionsState;

        if (key.return) {
            if (suggestions.length > 0 && selectedSuggestion >= 0) {
                handleSelection();
            }
            return;
        }

        if (key.tab && !key.shift) {
            if (suggestions.length > 0) {
                // Determine completion
                if (selectedSuggestion === -1) {
                    // Auto-select first?
                    setSuggestionsState((prev: any) => ({ ...prev, selectedSuggestion: 0 }));
                    // Then handle? or wait for next enter?
                    // Usually tab completes immediately if 1 match or cycles relative logic
                    // For now, let's just cycle or select first
                }
                handleSelection();
            }
            return;
        }

        if (suggestions.length === 0) return;

        if (key.downArrow || (key.ctrl && input === 'n')) {
            setSuggestionsState((prev: any) => ({
                ...prev,
                selectedSuggestion: (prev.selectedSuggestion >= prev.suggestions.length - 1) ? 0 : prev.selectedSuggestion + 1
            }));
            return;
        }

        if (key.upArrow || (key.ctrl && input === 'p')) {
            setSuggestionsState((prev: any) => ({
                ...prev,
                selectedSuggestion: (prev.selectedSuggestion <= 0) ? prev.suggestions.length - 1 : prev.selectedSuggestion - 1
            }));
            return;
        }

        if (key.escape) {
            clearSuggestions();
        }
    });


    return {
        ...suggestionsState,
        suggestionType,
        maxColumnWidth
    };
}

function adjustSelection(oldList: any[], oldIdx: number, newList: any[]) {
    if (oldIdx === -1) return -1;
    // Try to find the same item in new list
    const oldItem = oldList[oldIdx];
    if (!oldItem) return -1;
    const newIdx = newList.findIndex(i => i.displayText === oldItem.displayText);
    return newIdx !== -1 ? newIdx : -1;
}
