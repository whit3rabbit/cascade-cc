import React from "react";
import { Box } from "ink";
import { Text } from "../../vendor/inkText.js";

/**
 * A component to display a keyboard shortcut hint.
 */
export const Shortcut: React.FC<{
    shortcut: string;
    action: string;
    parens?: boolean;
    bold?: boolean;
}> = ({ shortcut, action, parens = false, bold = false }) => {
    const key = bold ? <Text bold>{shortcut}</Text> : shortcut;
    if (parens) {
        return <Text>({key} to {action})</Text>;
    }
    return <Text>{key} to {action}</Text>;
};

export const ShortcutHint = Shortcut;

export const ShortcutGroup: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    return <Box flexDirection="row" gap={2}>{children}</Box>;
};
