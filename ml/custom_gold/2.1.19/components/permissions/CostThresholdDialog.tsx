import React from 'react';
import { Box, Text, useInput } from 'ink';

export interface CostThresholdDialogProps {
    onApprove: () => void;
    onExit: () => void;
    cost: number;
    threshold: number;
}

export const CostThresholdDialog: React.FC<CostThresholdDialogProps> = ({ onApprove, onExit, cost, threshold }) => {
    useInput((input, key) => {
        if (input.toLowerCase() === 'y' || key.return) {
            onApprove();
        }
        if (input.toLowerCase() === 'n' || key.escape) {
            onExit();
        }
    });

    return (
        <Box flexDirection="column" borderStyle="double" borderColor="red" padding={1}>
            <Text bold color="red">⚠️ Cost Threshold Reached</Text>
            <Box marginY={1}>
                <Text>
                    This session has exceeded the cost threshold of <Text bold>${threshold.toFixed(2)}</Text>.
                    Current cost: <Text bold color="red">${cost.toFixed(2)}</Text>
                </Text>
            </Box>
            <Box flexDirection="column">
                <Text>Do you want to continue?</Text>
                <Box marginTop={1}>
                    <Text bold color="green"> (y) Yes, continue </Text>
                    <Text bold color="gray"> (n) No, exit </Text>
                </Box>
            </Box>
        </Box>
    );
};
