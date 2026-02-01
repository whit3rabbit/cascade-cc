/**
 * File: src/components/permissions/SandboxPermissionDialog.tsx
 * Role: Dialog for Sandbox Network Permissions
 */

import React from 'react';
import { Box, Text } from 'ink';

import { useInput } from 'ink';
import { PermissionOption } from './PermissionOption.js';
import { updateSettings } from '../../services/config/SettingsService.js';

export interface SandboxPermissionDialogProps {
    hostPattern: { host: string; port?: number };
    onDone: (result: 'allowed' | 'denied') => void;
}

export const SandboxPermissionDialog: React.FC<SandboxPermissionDialogProps> = ({ hostPattern, onDone }) => {
    const [selectedIndex, setSelectedIndex] = React.useState(0);
    const options = [
        { label: "Allow once", value: "once" },
        { label: "Always allow domain", value: "always" },
        { label: "Deny", value: "deny" }
    ];

    useInput((input, key) => {
        if (key.downArrow || input === 'j') {
            setSelectedIndex((prev) => (prev + 1) % options.length);
        }
        if (key.upArrow || input === 'k') {
            setSelectedIndex((prev) => (prev - 1 + options.length) % options.length);
        }
        if (key.return) {
            const selected = options[selectedIndex];
            if (selected.value === 'deny') {
                onDone('denied');
            } else if (selected.value === 'always') {
                // Persistent sandbox settings update
                const domain = hostPattern?.host;
                if (domain) {
                    updateSettings((current) => ({
                        ...current,
                        sandbox: {
                            ...current.sandbox,
                            enabled: current.sandbox?.enabled ?? true, // ensure it's still enabled
                            network: {
                                ...current.sandbox?.network,
                                allowedDomains: Array.from(new Set([...(current.sandbox?.network?.allowedDomains || []), domain]))
                            }
                        }
                    }));
                }
                onDone('allowed');
            } else {
                onDone('allowed');
            }
        }
        if (key.escape) {
            onDone('denied');
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="red" padding={1}>
            <Text bold color="red">Sandbox Network Access</Text>
            <Box marginY={1}>
                <Text>
                    An action is attempting to access <Text bold>{hostPattern?.host}</Text>.
                    Sandbox policy restricts network access.
                </Text>
            </Box>

            <Box flexDirection="column" marginTop={1}>
                {options.map((opt, i) => (
                    <PermissionOption
                        key={i}
                        label={opt.label}
                        value={opt.value as any}
                        isFocused={i === selectedIndex}
                        onSelect={() => { }} // key handler handles select
                        shortcut={i === 0 ? 'Enter' : undefined}
                    />
                ))}
            </Box>

            <Box marginTop={1}>
                <Text dimColor>Press Enter to confirm, Esc to reject</Text>
            </Box>
        </Box>
    );
};
