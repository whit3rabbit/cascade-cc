/**
 * File: src/components/menus/RemoteEnvMenu.tsx
 * Role: Menu for managing remote Claude.ai environments.
 */

import React from 'react';
import { Box, Text } from 'ink';
import SelectInput from 'ink-select-input';

interface RemoteEnvMenuProps {
    onExit: () => void;
}

export function RemoteEnvMenu({ onExit }: RemoteEnvMenuProps) {
    const items = [
        { label: 'Default (local)', value: 'local' },
        { label: 'Cloud Environment (AWS)', value: 'aws' },
        { label: 'Cloud Environment (GCP)', value: 'gcp' },
        { label: 'Managed Remote (Claude.ai)', value: 'managed' },
    ];

    const handleSelect = (_item: { value: string }) => {
        // Placeholder for environment switching logic
        onExit();
    };

    return (
        <Box flexDirection="column" paddingX={2}>
            <Box marginBottom={1}>
                <Text bold underline>Configure Remote Environment</Text>
            </Box>
            <Text>Select the default remote environment for teleport sessions:</Text>
            <Box marginTop={1}>
                <SelectInput items={items} onSelect={handleSelect} />
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Esc to cancel · Enter to select</Text>
            </Box>
        </Box>
    );
}

export default RemoteEnvMenu;
