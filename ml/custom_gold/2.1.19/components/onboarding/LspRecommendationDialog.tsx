/**
 * File: src/components/onboarding/LspRecommendationDialog.tsx
 * Role: Dialog for LSP Recommendations
 */

import React from 'react';
import { Box, Text } from 'ink';

export const LspRecommendationDialog: React.FC<any> = ({ recommendation, onAccept, onDecline }) => {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor="green">
            <Text bold color="green">Language Server Recommendation</Text>
            <Text>We detected a {recommendation?.language} project.</Text>
            <Text>Install recommended LSP?</Text>
            <Text color="gray">Implementation pending...</Text>
        </Box>
    );
};
