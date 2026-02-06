/**
 * File: src/components/menus/LoopMenu.tsx
 * Role: Interactive menu for controlling agent loops.
 */

import React from 'react';
import { Box, Text } from 'ink';
import SelectInput from 'ink-select-input';
import { useTheme } from '../../services/terminal/ThemeService.js';

interface LoopMenuProps {
    onExit: () => void;
}

export const LoopMenu: React.FC<LoopMenuProps> = ({ onExit }) => {
    const theme = useTheme();

    const handleSelect = (item: { value: string }) => {
        if (item.value === 'exit') {
            onExit();
        }
        // Additional loop control logic (Stop/Pause) would go here
    };

    return (
        <Box flexDirection="column" padding={1} borderStyle="round" borderColor={theme.claudeBlue_FOR_SYSTEM_SPINNER}>
            <Text bold color={theme.claudeBlue_FOR_SYSTEM_SPINNER}>Control Loop</Text>
            <Box marginTop={1}>
                <SelectInput
                    items={[
                        { label: 'Stop Loop', value: 'stop' },
                        { label: 'Pause Loop', value: 'pause' },
                        { label: 'Exit Menu', value: 'exit' }
                    ]}
                    onSelect={handleSelect}
                />
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Esc to exit</Text>
            </Box>
        </Box>
    );
};
