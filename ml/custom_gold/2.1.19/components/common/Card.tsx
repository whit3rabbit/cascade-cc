/**
 * File: src/components/common/Card.tsx
 * Role: Reusable stylized Card component for Ink-based terminal UIs.
 */

import React from 'react';
import { Box, Text } from 'ink';

interface CardProps {
    title?: string;
    subtitle?: string;
    children: React.ReactNode;
    borderColor?: string;
    titleColor?: string;
    paddingX?: number;
    paddingY?: number;
    marginBottom?: number;
}

/**
 * A stylized card component with a title, optional subtitle, and child content.
 * Designed to provide consistent visual grouping in terminal applications.
 */
export function Card({
    title,
    subtitle,
    children,
    borderColor = "cyan",
    titleColor = "white",
    paddingX = 1,
    paddingY = 0,
    marginBottom = 1
}: CardProps) {
    return (
        <Box
            flexDirection="column"
            borderStyle="round"
            borderColor={borderColor}
            paddingX={paddingX}
            paddingY={paddingY}
            marginBottom={marginBottom}
        >
            {title && (
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold color={titleColor}>{title.toUpperCase()}</Text>
                    {subtitle && <Text dimColor>{subtitle}</Text>}
                </Box>
            )}
            <Box flexDirection="column">
                {children}
            </Box>
        </Box>
    );
}

export default Card;
