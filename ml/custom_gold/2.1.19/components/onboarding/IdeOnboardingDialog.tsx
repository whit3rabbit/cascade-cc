/**
 * File: src/components/onboarding/IdeOnboardingDialog.tsx
 * Role: Dialog for IDE Setup
 */

import React from 'react';
import { Box, Text } from 'ink';

export const IdeOnboardingDialog: React.FC<any> = ({ onDone }) => {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor="blue">
            <Text bold color="blue">IDE Setup</Text>
            <Text>Would you like to configure your IDE integration?</Text>
            <Text color="gray">Implementation pending...</Text>
        </Box>
    );
};
