import * as React from "react";
import { Box } from "../../vendor/inkBox.js";
import { Text } from "../../vendor/inkText.js";
import { figures } from "../../vendor/terminalFigures.js";

interface SelectItemProps {
    isFocused: boolean;
    isSelected?: boolean;
    children: React.ReactNode;
    description?: string;
    shouldShowDownArrow?: boolean;
    shouldShowUpArrow?: boolean;
}

/**
 * Component for rendering items in a select list.
 * Deobfuscated from yi in chunk_205.ts.
 */
export const SelectItem: React.FC<SelectItemProps> = ({
    isFocused,
    isSelected,
    children,
    description,
    shouldShowDownArrow,
    shouldShowUpArrow
}) => {
    return (
        <Box flexDirection="column">
            <Box flexDirection="row" gap={1}>
                {isFocused ? (
                    <Text color="suggestion">{figures.pointer}</Text>
                ) : shouldShowDownArrow ? (
                    <Text dimColor={true}>{figures.arrowDown}</Text>
                ) : shouldShowUpArrow ? (
                    <Text dimColor={true}>{figures.arrowUp}</Text>
                ) : (
                    <Text>{" "}</Text>
                )}
                {children}
                {isSelected && <Text color="success">{figures.tick}</Text>}
            </Box>
            {description && (
                <Box paddingLeft={5}>
                    <Text color="inactive">{description}</Text>
                </Box>
            )}
        </Box>
    );
};
