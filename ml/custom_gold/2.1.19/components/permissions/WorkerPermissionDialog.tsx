/**
 * File: src/components/permissions/WorkerPermissionDialog.tsx
 * Role: Dialog for Worker Permissions
 */

import React from 'react';
import { Box, Text } from 'ink';

export const WorkerPermissionDialog: React.FC<any> = ({ request, onDone, onApprove }) => {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor="magenta">
            <Text bold color="magenta">Worker Permission Required</Text>
            <Text>Allow worker {request?.workerName}?</Text>
            <Text color="gray">Implementation pending...</Text>
        </Box>
    );
};
