
// Logic from chunk_567.ts / chunk_593.ts (Interactive Select)

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";

export interface Option {
    label: string;
    value: string;
}

export interface SelectProps {
    options: Option[];
    onChange: (value: string) => void;
    onCancel?: () => void;
}

export function Select({ options, onChange, onCancel }: SelectProps) {
    const [selectedIndex, setSelectedIndex] = useState(0);

    useInput((input, key) => {
        if (key.upArrow) {
            setSelectedIndex(i => Math.max(0, i - 1));
        } else if (key.downArrow) {
            setSelectedIndex(i => Math.min(options.length - 1, i + 1));
        } else if (key.return) {
            onChange(options[selectedIndex].value);
        } else if (key.escape && onCancel) {
            onCancel();
        }
    });

    return (
        <Box flexDirection="column">
            {options.map((option, index) => {
                const isSelected = index === selectedIndex;
                return (
                    <Box key={option.value}>
                        <Text color={isSelected ? "claude" : undefined}>
                            {isSelected ? figures.pointer : " "} {option.label}
                        </Text>
                    </Box>
                );
            })}
        </Box>
    );
}
