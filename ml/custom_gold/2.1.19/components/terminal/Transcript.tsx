import React from 'react';
import { Box, Text } from 'ink';
import { MessageRenderer } from './MessageRenderer.js';

export interface TranscriptProps {
    messages: any[];
    scrollOffset?: number;
    showAll?: boolean;
    onToggleShowAll?: () => void;
    rows?: number;
    columns?: number;
}

const TRUNCATION_THRESHOLD = 20;
const MIN_ROWS_FOR_INPUT = 5;

/**
 * Overall message history container, mirroring OI2 from golden source.
 * Handles windowed rendering (virtualization) based on terminal height.
 */
export const Transcript: React.FC<TranscriptProps> = ({
    messages,
    scrollOffset = 0,
    showAll = false,
    onToggleShowAll,
    rows = 24,
    columns = 80
}) => {
    // Determine how many messages to show based on terminal height
    // We leave some space for the header, input area, and status line
    const availableHeight = Math.max(5, rows - MIN_ROWS_FOR_INPUT - 4); // basic heuristic

    // For now, we still use a threshold but make it somewhat responsive
    const threshold = showAll ? messages.length : Math.min(TRUNCATION_THRESHOLD, Math.floor(availableHeight * 1.5));

    const hasTruncatedMessages = messages.length > threshold;
    const visibleMessages = hasTruncatedMessages ? messages.slice(-threshold) : messages;

    return (
        <Box
            flexDirection="column"
            flexGrow={1}
            overflowY="hidden"
            width="100%"
            height={availableHeight}
        >
            <Box flexDirection="column" marginTop={0}>
                {hasTruncatedMessages && !showAll && (
                    <Box borderStyle="single" borderColor="gray" paddingX={1} marginY={1} justifyContent="center">
                        <Text dimColor>
                            --- Showing last {threshold} messages. Press Ctrl+E to show all ---
                        </Text>
                    </Box>
                )}

                {visibleMessages.map((msg, index) => (
                    <MessageRenderer
                        key={msg.uuid || index}
                        message={msg}
                    />
                ))}

                {showAll && messages.length > TRUNCATION_THRESHOLD && (
                    <Box borderStyle="single" borderColor="gray" paddingX={1} marginY={1} justifyContent="center">
                        <Text dimColor>
                            --- Showing all messages. Press Ctrl+E to hide older messages ---
                        </Text>
                    </Box>
                )}
            </Box>
        </Box>
    );
};
