/**
 * File: src/components/terminal/TranscriptToggle.tsx
 * Role: UI component for toggling the detailed command transcript visibility.
 */

import React from 'react';
import { Box, Text } from 'ink';

interface TranscriptToggleProps {
    shortcut?: string;
    isActive?: boolean;
}

/**
 * Renders a footer hint indicating that the transcript view is active and provide the toggle shortcut.
 */
export function TranscriptToggle({ shortcut = "Ctrl+O", isActive = false }: TranscriptToggleProps) {
    if (!isActive) return null;

    return (
        <Box
            alignItems="center"
            borderStyle="single"
            borderTop={true}
            borderBottom={false}
            borderLeft={false}
            borderRight={false}
            borderColor="gray"
            marginTop={1}
            paddingLeft={1}
            width="100%"
        >
            <Text dimColor>
                Showing detailed transcript · <Text bold color="yellow">{shortcut}</Text> to toggle
            </Text>
        </Box>
    );
}

export default TranscriptToggle;
