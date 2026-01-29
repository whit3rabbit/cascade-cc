/**
 * File: src/components/common/ActionHint.tsx
 * Role: Standardized UI component for displaying keyboard shortcuts and actions.
 */

import React from 'react';
import { Box, Text } from 'ink';

interface ActionHintProps {
    shortcut: string;
    action: string;
    dimColor?: boolean;
    bold?: boolean;
    color?: string;
}

/**
 * Renders a keyboard shortcut hint, e.g., "[Enter] confirm".
 */
export function ActionHint({
    shortcut,
    action,
    dimColor = true,
    bold = false,
    color = "white"
}: ActionHintProps) {
    return (
        <Box marginRight={2}>
            <Text dimColor={dimColor} bold={bold} color={color}>
                [{shortcut}]
            </Text>
            <Text> {action}</Text>
        </Box>
    );
}

export default ActionHint;
