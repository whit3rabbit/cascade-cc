import React from 'react';
import { useInput } from 'ink';

export interface REPLKeybindingsProps {
    onCancel: () => void;
    screen: string;
    vimMode: 'INSERT' | 'NORMAL';
    isSearchingHistory: boolean;
    isHelpOpen: boolean;
    inputMode: string;
    inputValue: string;
    // ... other props from golden source VX object
}

/**
 * REPLKeybindings equivalent to XF6 in golden source.
 * Handles global shortcuts that are active across the whole session.
 */
export const REPLKeybindings: React.FC<REPLKeybindingsProps> = ({
    onCancel,
    vimMode: _vimMode
}) => {
    useInput((input, key) => {
        // Porting logic from current REPL.tsx global useInput
        if (input === 'c' && key.ctrl) {
            // This is usually handled by useExitOnCtrlCD hook in golden source
            // but we can put it here or in a specialized hook.
            onCancel();
        }

        // Handle other global keys like Ctrl+L (clear), Ctrl+T (tasks), etc.
    });

    return null; // Logic-only component
};
