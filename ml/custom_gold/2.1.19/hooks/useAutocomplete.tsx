/**
 * File: src/hooks/useAutocomplete.tsx
 * Role: Provides autocomplete logic for slash commands, files, and directories.
 */

import { useState, useCallback, useRef, useEffect, Dispatch, SetStateAction } from 'react';
import { useInput } from 'ink';
import {
    suggestSlashCommands,
    suggestFiles,
    suggestDirectories
} from '../services/terminal/FileIndexService.js';
import { Suggestion } from '../services/terminal/FileIndexService.js';
import { getBashHistory } from '../utils/terminal/bashHistory.js';

const DEBOUNCE_DELAY_MS = 200;

export interface SuggestionsState {
    commandArgumentHint?: string;
    suggestions: Suggestion[];
    selectedSuggestionIndex: number;
}

export interface UseAutocompleteProps {
    commands: any[];
    onInputChange: (newInput: string) => void;
    setCursorOffset: (newOffset: number) => void;
    input: string;
    cursorOffset: number;
    mode: string;
    suggestionsState: SuggestionsState;
    setSuggestionsState: Dispatch<SetStateAction<SuggestionsState>>;
    suppressSuggestions?: boolean;
}

/**
 * Hook for managing autocomplete state and interactions.
 */
export function useAutocomplete({
    commands,
    onInputChange,
    setCursorOffset,
    input,
    cursorOffset,
    mode,
    suggestionsState,
    setSuggestionsState,
    suppressSuggestions = false
}: UseAutocompleteProps): null {
    const [, setSuggestionSourceType] = useState<string>("none");
    const lastFetchInputRef = useRef<string>("");

    const clearSuggestions = useCallback(() => {
        setSuggestionsState({
            commandArgumentHint: undefined,
            suggestions: [],
            selectedSuggestionIndex: -1
        });
        setSuggestionSourceType("none");
    }, [setSuggestionsState]);

    /**
     * Updates the suggestions based on the current input context and cursor position.
     */
    const performSuggestionsUpdate = useCallback(async (text: string, offset: number) => {
        if (suppressSuggestions) {
            clearSuggestions();
            return;
        }

        lastFetchInputRef.current = text;
        const snapshotInput = text;

        const textBeforeCursor = text.substring(0, offset);
        const isTrailingSpace = offset === text.length && offset > 0 && text[offset - 1] === ' ';

        // 1. Directory completion for /add-dir command
        if (mode === 'prompt' && text.startsWith('/add-dir ') && !isTrailingSpace) {
            const pathPrefix = text.substring('/add-dir '.length);
            const dirs = await suggestDirectories(pathPrefix);
            if (snapshotInput !== lastFetchInputRef.current) return;

            if (dirs.length > 0) {
                setSuggestionsState(prev => ({
                    ...prev,
                    suggestions: dirs,
                    selectedSuggestionIndex: calculateNextSelectedIndex(prev.suggestions, prev.selectedSuggestionIndex, dirs)
                }));
                setSuggestionSourceType('directory');
                return;
            }
        }

        // 2. Slash command completion
        if (mode === 'prompt' && text.startsWith('/') && !text.includes(' ') && !isTrailingSpace) {
            const suggestions = await suggestSlashCommands(text, commands);
            if (snapshotInput !== lastFetchInputRef.current) return;

            setSuggestionsState(prev => ({
                ...prev,
                suggestions,
                selectedSuggestionIndex: calculateNextSelectedIndex(prev.suggestions, prev.selectedSuggestionIndex, suggestions)
            }));
            setSuggestionSourceType(suggestions.length > 0 ? 'command' : 'none');
            return;
        }

        // 3. File completion (triggered by '@')
        const atRegexMatch = textBeforeCursor.match(/(^|\s)@([a-zA-Z0-9_\-./\\()[\]~]*)$/);
        if (atRegexMatch) {
            const fileQuery = atRegexMatch[2];
            const files = await suggestFiles(fileQuery);
            if (snapshotInput !== lastFetchInputRef.current) return;

            if (files.length > 0) {
                setSuggestionsState(prev => ({
                    ...prev,
                    suggestions: files,
                    selectedSuggestionIndex: calculateNextSelectedIndex(prev.suggestions, prev.selectedSuggestionIndex, files)
                }));
                setSuggestionSourceType('file');
                return;
            }
        }

        // 4. Shell history completion (if nothing else matched)
        if (mode === 'prompt' && text.length >= 3) {
            const history = await getBashHistory();
            if (snapshotInput !== lastFetchInputRef.current) return;

            const filteredHistory = history
                .filter(cmd => cmd.toLowerCase().includes(text.toLowerCase()))
                .slice(0, 10)
                .map((cmd, idx) => ({
                    id: `history-${idx}`,
                    displayText: cmd,
                    description: 'from shell history'
                }));

            if (filteredHistory.length > 0) {
                setSuggestionsState(prev => ({
                    ...prev,
                    suggestions: filteredHistory,
                    selectedSuggestionIndex: calculateNextSelectedIndex(prev.suggestions, prev.selectedSuggestionIndex, filteredHistory)
                }));
                setSuggestionSourceType('history');
                return;
            }
        }

        clearSuggestions();
    }, [mode, commands, suppressSuggestions, clearSuggestions, setSuggestionsState]);

    // Debounce the suggestion updates to avoid excessive processing
    useEffect(() => {
        const debounceTimer = setTimeout(() => {
            performSuggestionsUpdate(input, cursorOffset);
        }, DEBOUNCE_DELAY_MS);
        return () => clearTimeout(debounceTimer);
    }, [input, cursorOffset, performSuggestionsUpdate]);

    /**
     * Applies the selected suggestion to the current terminal input.
     */
    const applySelectedSuggestion = useCallback(() => {
        const { suggestions, selectedSuggestionIndex } = suggestionsState;
        if (selectedSuggestionIndex < 0 || suggestions.length === 0) return;

        const selectedItem = suggestions[selectedSuggestionIndex];
        const textBeforeCursor = input.substring(0, cursorOffset);

        // Find the start of the token being completed
        const parts = textBeforeCursor.split(/\s+/);
        const lastPart = parts[parts.length - 1] || "";
        const prefix = textBeforeCursor.substring(0, textBeforeCursor.length - lastPart.length);
        const suffix = input.substring(cursorOffset);

        const replacementText = selectedItem.displayText;
        const newFullText = prefix + replacementText + " " + suffix;

        onInputChange(newFullText);
        setCursorOffset(prefix.length + replacementText.length + 1);
        clearSuggestions();
    }, [suggestionsState, input, cursorOffset, onInputChange, setCursorOffset, clearSuggestions]);

    // Intercept keyboard input for navigation and selection
    useInput((_, key) => {
        const { suggestions } = suggestionsState;
        if (suggestions.length === 0) return;

        if (key.return || key.tab) {
            applySelectedSuggestion();
            return;
        }

        if (key.downArrow) {
            setSuggestionsState(prev => ({
                ...prev,
                selectedSuggestionIndex: (prev.selectedSuggestionIndex + 1) % suggestions.length
            }));
        }

        if (key.upArrow) {
            setSuggestionsState(prev => ({
                ...prev,
                selectedSuggestionIndex: (prev.selectedSuggestionIndex - 1 < 0)
                    ? suggestions.length - 1
                    : prev.selectedSuggestionIndex - 1
            }));
        }

        if (key.escape) {
            clearSuggestions();
        }
    });

    return null;
}

/**
 * Calculates the next selected index, trying to keep the same item selected if possible.
 */
function calculateNextSelectedIndex(currentItems: Suggestion[], currentIndex: number, newItems: Suggestion[]): number {
    if (!newItems || newItems.length === 0) return -1;
    if (currentItems && currentIndex >= 0 && currentIndex < currentItems.length) {
        const currentlySelectedText = currentItems[currentIndex].displayText;
        const newMatchIndex = newItems.findIndex(item => item.displayText === currentlySelectedText);
        if (newMatchIndex !== -1) return newMatchIndex;
    }
    return 0;
}
