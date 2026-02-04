import React from 'react';
import { Box } from 'ink';

export interface REPLInputProps {
    onSubmit: (value: string) => void;
    isActive: boolean;
    children?: React.ReactNode;
}

/**
 * REPLInput equivalent to JF6 in golden source.
 * A wrapper for the input part of the REPL.
 */
export const REPLInput: React.FC<REPLInputProps> = ({
    isActive,
    children
}) => {
    if (!isActive) {
        return null;
    }

    return (
        <Box flexDirection="column" width="100%">
            {children}
        </Box>
    );
};
