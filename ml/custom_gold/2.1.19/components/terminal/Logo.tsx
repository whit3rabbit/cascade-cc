/**
 * File: src/components/terminal/Logo.tsx
 * Role: Renders the official Claude Code ASCII logo.
 */

import React from 'react';
import { Box, Text } from 'ink';
import { THEME } from '../../constants/theme.js';

interface LogoProps {
    version: string;
    model: string;
    cwd: string;
}

export const Logo: React.FC<LogoProps> = ({ version, model, cwd }) => {
    // Colors from theme
    const bodyColor = THEME.colors.clawd_body;
    const bgColor = THEME.colors.clawd_background;

    // In a real implementation this would come from a billing context
    const billing = "Claude API";

    // Reconstruction of the ASCII art from chunk1339
    // Row 1:  ▐▛███▜▌
    // Row 2: ▝▜█████▛▘
    // Row 3:   ▘▘ ▝▝

    return (
        <Box flexDirection="row" gap={2} alignItems="center" paddingY={1}>
            {/* The Icon */}
            <Box flexDirection="column">
                <Box>
                    <Text color={bodyColor}> ▐</Text>
                    <Text color={bodyColor} backgroundColor={bgColor}>▛███▜</Text>
                    <Text color={bodyColor}>▌</Text>
                </Box>
                <Box>
                    <Text color={bodyColor}>▝▜</Text>
                    <Text color={bodyColor} backgroundColor={bgColor}>█████</Text>
                    <Text color={bodyColor}>▛▘</Text>
                </Box>
                <Box>
                    <Text>  </Text>
                    <Text color={bodyColor}>▘▘ ▝▝</Text>
                    <Text>  </Text>
                </Box>
            </Box>

            {/* The Text Info */}
            <Box flexDirection="column">
                <Box>
                    <Text bold>Claude Code</Text>
                    <Text dimColor> v{version}</Text>
                </Box>
                <Box>
                    <Text dimColor>{model} · {billing}</Text>
                </Box>
                <Box>
                    <Text dimColor>{cwd}</Text>
                </Box>
            </Box>
        </Box>
    );
};
