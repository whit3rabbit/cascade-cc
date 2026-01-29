/**
 * File: src/components/common/Divider.tsx
 * Role: A simple horizontal divider for terminal UI.
 */

import React from 'react';
import { Box, Text } from 'ink';

interface DividerProps {
    char?: string;
    color?: string;
    dimColor?: boolean;
    marginBottom?: number;
}

/**
 * Renders a horizontal line across the terminal.
 */
export function Divider({
    char = "─",
    color = "gray",
    dimColor = true,
    marginBottom = 0
}: DividerProps) {
    return (
        <Box width="100%" height={1} overflow="hidden" marginBottom={marginBottom}>
            <Text color={color} dimColor={dimColor}>
                {char.repeat(100)}
            </Text>
        </Box>
    );
}

export default Divider;
