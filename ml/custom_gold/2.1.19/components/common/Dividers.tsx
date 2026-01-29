/**
 * File: src/components/common/Dividers.tsx
 * Role: Horizontal divider components for terminal UI.
 */

import React from 'react';
import { Box, Text } from 'ink';

const DEFAULT_LINE_LENGTH = 80;

/**
 * Simple horizontal divider line.
 */
export const Divider: React.FC = () => (
    <Box paddingX={1}>
        <Text dimColor>{"─".repeat(DEFAULT_LINE_LENGTH)}</Text>
    </Box>
);

/**
 * Horizontal divider with a title.
 */
export const TitledDivider: React.FC<{ title: string }> = ({ title }) => {
    const titleLength = title.length + 2;
    const sideLength = Math.max(0, Math.floor((DEFAULT_LINE_LENGTH - titleLength) / 2));
    const line = "─".repeat(sideLength);

    return (
        <Box paddingX={1}>
            <Text dimColor>{line} </Text>
            <Text bold>{title}</Text>
            <Text dimColor> {line}</Text>
        </Box>
    );
};
