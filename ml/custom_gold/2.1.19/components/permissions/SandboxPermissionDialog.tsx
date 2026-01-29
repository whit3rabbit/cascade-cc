/**
 * File: src/components/permissions/SandboxPermissionDialog.tsx
 * Role: Dialog for Sandbox Network Permissions
 */

import React from 'react';
import { Box, Text } from 'ink';

export const SandboxPermissionDialog: React.FC<any> = ({ hostPattern, onUserResponse }) => {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor="red">
            <Text bold color="red">Sandbox Network Access</Text>
            <Text>Allow access to {hostPattern?.host}?</Text>
            <Text color="gray">Implementation pending...</Text>
        </Box>
    );
};
