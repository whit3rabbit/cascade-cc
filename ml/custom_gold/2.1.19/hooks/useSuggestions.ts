import { useState, useCallback, useMemo } from 'react';

export interface Suggestion {
    id: string;
    displayText: string;
    description?: string;
}

export interface UseSuggestionsOptions {
    input: string;
    commands: any[];
    agents?: any[];
}

/**
 * Simplified hook for input suggestions, inspired by tfK from golden source.
 */
export function useSuggestions({
    input,
    commands,
    agents = []
}: UseSuggestionsOptions) {
    const [selectedIndex, setSelectedIndex] = useState(-1);

    const suggestions = useMemo(() => {
        if (!input.startsWith('/')) return [];

        const partial = input.slice(1).toLowerCase();
        const cmdSuggestions = commands
            .filter(c => c.name.toLowerCase().startsWith(partial))
            .map(c => ({
                id: `cmd-${c.name}`,
                displayText: `/${c.name}`,
                description: c.description
            }));

        const agentSuggestions = agents
            .filter(a => a.name.toLowerCase().startsWith(partial))
            .map(a => ({
                id: `agent-${a.name}`,
                displayText: `/${a.name}`,
                description: "agent"
            }));

        return [...cmdSuggestions, ...agentSuggestions];
    }, [input, commands, agents]);

    const selectedSuggestion = useMemo(() => {
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
            return suggestions[selectedIndex];
        }
        return undefined;
    }, [suggestions, selectedIndex]);

    const onNextSuggestion = useCallback(() => {
        setSelectedIndex(prev =>
            suggestions.length > 0 ? (prev + 1) % suggestions.length : -1
        );
    }, [suggestions.length]);

    const onPrevSuggestion = useCallback(() => {
        setSelectedIndex(prev =>
            suggestions.length > 0 ? (prev - 1 + suggestions.length) % suggestions.length : -1
        );
    }, [suggestions.length]);

    const resetSuggestions = useCallback(() => {
        setSelectedIndex(-1);
    }, []);

    return {
        suggestions,
        selectedSuggestion,
        selectedIndex,
        onNextSuggestion,
        onPrevSuggestion,
        resetSuggestions
    };
}
