/**
 * File: src/components/menus/UnifiedSearchMenu.tsx
 * Role: Searchable command palette for history and slash commands.
 */

import React, { useState, useMemo } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import SelectInput from 'ink-select-input';
import { useTheme } from '../../services/terminal/ThemeService.js';

interface UnifiedSearchMenuProps {
    history: string[];
    commands: { label: string; value: string; description?: string }[];
    onSelect: (value: string) => void;
    onExit: () => void;
}

export const UnifiedSearchMenu: React.FC<UnifiedSearchMenuProps> = ({ history, commands, onSelect, onExit }) => {
    const theme = useTheme();
    const [query, setQuery] = useState('');

    // Combine items
    const allItems = useMemo(() => {
        const historyItems = [...new Set(history)].reverse().map(h => ({
            label: h,
            value: h,
            type: 'history'
        }));

        const commandItems = commands.map(c => ({
            label: c.label,
            value: c.value,
            type: 'command',
            description: c.description
        }));

        return [...commandItems, ...historyItems];
    }, [history, commands]);

    // Filter items
    const filteredItems = useMemo(() => {
        if (!query) return allItems.slice(0, 20);
        const lowerQuery = query.toLowerCase();
        return allItems
            .filter(item => item.label.toLowerCase().includes(lowerQuery))
            .slice(0, 20);
    }, [allItems, query]);

    useInput((input, key) => {
        if (key.escape) onExit();
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor={theme.claudeBlue_FOR_SYSTEM_SPINNER} paddingX={1} width={60}>
            <Box marginBottom={1}>
                <Text bold color={theme.claudeBlue_FOR_SYSTEM_SPINNER}>Search: </Text>
                <TextInput value={query} onChange={setQuery} focus />
            </Box>

            <Box flexDirection="column">
                <SelectInput
                    items={filteredItems.map(item => ({
                        label: `${item.type === 'command' ? '[CMD]' : '[HIS]'} ${item.label}`,
                        value: item.value
                    }))}
                    onSelect={(item) => onSelect(item.value)}
                />
            </Box>

            {filteredItems.length === 0 && (
                <Text dimColor italic>No results found</Text>
            )}

            <Box marginTop={1}>
                <Text dimColor>Esc to close • Enter to select</Text>
            </Box>
        </Box>
    );
};
