/**
 * File: src/components/permissions/WorkerPermissionDialog.tsx
 * Role: Dialog for Worker Permissions
 */

import React from 'react';
import { Box, Text } from 'ink';

import { useInput } from 'ink';
import { PermissionOption } from './PermissionOption.js';

export interface WorkerPermissionDialogProps {
    request: { workerName: string; reason?: string };
    onApprove: () => void;
    onReject: () => void;
}

export const WorkerPermissionDialog: React.FC<WorkerPermissionDialogProps> = ({ request, onApprove, onReject }) => {
    const [selectedIndex, setSelectedIndex] = React.useState(0);
    const options = [
        { label: "Approve Worker", value: "approve" },
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
            if (selected.value === 'approve') {
                onApprove();
            } else {
                onReject();
            }
        }
        if (key.escape) {
            onReject();
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="magenta" padding={1}>
            <Text bold color="magenta">Worker Permission Required</Text>
            <Box marginY={1}>
                <Text>
                    Worker <Text bold>{request?.workerName}</Text> is requesting permission to join/execute.
                </Text>
                {request?.reason && <Text dimColor>{request.reason}</Text>}
            </Box>

            <Box flexDirection="column" marginTop={1}>
                {options.map((opt, i) => (
                    <PermissionOption
                        key={i}
                        label={opt.label}
                        value={opt.value as any}
                        isFocused={i === selectedIndex}
                        onSelect={() => { }}
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
