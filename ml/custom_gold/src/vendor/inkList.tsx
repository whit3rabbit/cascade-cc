import * as React from "react";
import { Box, Text } from "ink";

/**
 * Numbered list components for Ink.
 * Deobfuscated from fDB and GA1 in chunk_215.ts.
 */

const ListContext = React.createContext({ marker: "" });

export interface ListItemProps {
    children: React.ReactNode;
}

export function ListItem({ children }: ListItemProps) {
    const { marker } = React.useContext(ListContext);
    return (
        <Box gap={1}>
            <Text dimColor>{marker}</Text>
            <Box flexDirection="column">{children}</Box>
        </Box>
    );
}

export interface NumberedListProps {
    children: React.ReactNode;
}

export function NumberedList({ children }: NumberedListProps) {
    const { marker: parentMarker } = React.useContext(ListContext);

    const items = React.Children.toArray(children).filter(
        child => React.isValidElement(child) && child.type === ListItem
    );

    const maxLabelWidth = String(items.length).length;

    return (
        <Box flexDirection="column">
            {React.Children.map(children, (child, index) => {
                if (!React.isValidElement(child) || child.type !== ListItem) {
                    return child;
                }

                const label = `${String(index + 1).padStart(maxLabelWidth)}.`;
                const marker = `${parentMarker}${label}`;

                return (
                    <ListContext.Provider value={{ marker }}>
                        {child}
                    </ListContext.Provider>
                );
            })}
        </Box>
    );
}

NumberedList.Item = ListItem;
