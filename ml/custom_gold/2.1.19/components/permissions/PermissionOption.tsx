import React from 'react';
import { Box, Text } from 'ink';

export interface PermissionOptionProps {
    label: string;
    value: string;
    isFocused: boolean;
    onSelect: () => void;
    shortcut?: string;
    description?: string;
}

export const PermissionOption: React.FC<PermissionOptionProps> = ({
    label,
    value,
    isFocused,
    onSelect,
    shortcut,
    description
}) => {
    return (
        <Box flexDirection="column" paddingLeft={1}>
            <Box>
                <Text color={isFocused ? "cyan" : "gray"}>
                    {isFocused ? "> " : "  "}
                </Text>
                <Text bold={isFocused} color={isFocused ? "white" : "gray"}>
                    {label}
                </Text>
                {shortcut && (
                    <Text color="gray"> ({shortcut})</Text>
                )}
            </Box>
            {isFocused && description && (
                <Box paddingLeft={2}>
                    <Text color="gray" dimColor>{description}</Text>
                </Box>
            )}
        </Box>
    );
};
