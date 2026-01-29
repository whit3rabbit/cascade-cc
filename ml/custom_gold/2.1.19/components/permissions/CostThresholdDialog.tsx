/**
 * File: src/components/permissions/CostThresholdDialog.tsx
 * Role: Dialog for Cost Threshold warnings
 */

import React from 'react';
import { Box, Text } from 'ink';

export const CostThresholdDialog: React.FC<any> = ({ onApprove, onExit }) => {
    return (
        <Box flexDirection="column" borderStyle="double" borderColor="red">
            <Text bold color="red">Cost Threshold Reached</Text>
            <Text>You have exceeded the session cost limit.</Text>
            <Text>Continue?</Text>
            <Text color="gray">Implementation pending...</Text>
        </Box>
    );
};
