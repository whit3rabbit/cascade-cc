/**
 * File: src/components/terminal/Logo.tsx
 * Role: Renders the official Claude Code ASCII logo.
 */

import React from 'react';
import { Box, Text } from 'ink';
import { useTheme } from '../../services/terminal/ThemeService.js';
import { EnvService } from '../../services/config/EnvService.js';
import os from 'os';

interface LogoProps {
    version: string;
    model: string;
    cwd: string;
    subscription?: string;
}

export const Logo: React.FC<LogoProps> = ({ version, model, cwd, subscription }) => {
    const theme = useTheme();
    // Colors from theme
    const bodyColor = theme.clawd_body;
    const bgColor = theme.clawd_background;


    // Shorten cwd
    const home = os.homedir();
    const displayCwd = cwd.startsWith(home) ? cwd.replace(home, '~') : cwd;

    const billing = subscription || "Not Authenticated";

    const isAppleTerminal = EnvService.get("TERM_PROGRAM") === 'Apple_Terminal';

    const renderIcon = () => {
        if (isAppleTerminal) {
            return (
                <Box flexDirection="column" alignItems="center">
                    <Box>
                        <Text color={bodyColor}>▗</Text>
                        <Text color={bgColor} backgroundColor={bodyColor}> ▗   ▖ </Text>
                        <Text color={bodyColor}>▖</Text>
                    </Box>
                    <Text backgroundColor={bodyColor}>       </Text>
                    <Text color={bodyColor}>▘▘ ▝▝</Text>
                </Box>
            );
        }

        return (
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
                    <Text color={bodyColor}>  ▘▘ ▝▝  </Text>
                </Box>
            </Box>
        );
    };

    return (
        <Box flexDirection="row" gap={2} alignItems="center" paddingY={1}>
            {/* The Icon */}
            {renderIcon()}

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
                    <Text dimColor>{displayCwd}</Text>
                </Box>
            </Box>
        </Box>
    );
};
