/**
 * File: src/components/common/Select.tsx
 * Role: A standardized SelectInput wrapper for Ink terminal UI.
 */

import React from 'react';
import { Box, Text } from 'ink';
import SelectInput from 'ink-select-input';

interface Option {
    label: string;
    value: string;
    description?: string;
}

interface SelectProps {
    options: Option[];
    defaultValue?: string;
    onChange: (value: string) => void;
    onFocus?: (value: string) => void;
    onCancel?: () => void;
    visibleOptionCount?: number;
    title?: string;
    emptyMessage?: string;
}

/**
 * Standardized Select component that handles list navigation and selection.
 */
export function Select({
    options,
    defaultValue,
    onChange,
    onFocus,
    onCancel,
    visibleOptionCount = 10,
    title,
    emptyMessage = "No options available"
}: SelectProps) {
    if (options.length === 0) {
        return <Text dimColor>{emptyMessage}</Text>;
    }

    return (
        <Box flexDirection="column" marginBottom={1}>
            {title && <Text bold>{title}</Text>}
            <SelectInput
                items={options}
                initialIndex={defaultValue ? options.findIndex(o => o.value === defaultValue) : 0}
                onSelect={(item) => onChange(item.value)}
                onHighlight={(item) => onFocus?.(item.value)}
                limit={visibleOptionCount}
            />
        </Box>
    );
}

export default Select;
