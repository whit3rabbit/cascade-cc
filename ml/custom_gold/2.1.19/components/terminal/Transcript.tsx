import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { MessageRenderer } from './MessageRenderer.js';
import { MessageHeightCalculator } from '../../utils/terminal/MessageHeightCalculator.js';
import { MeasuredText } from '../../utils/terminal/MeasuredText.js';

export interface TranscriptProps {
    messages: any[];
    scrollOffset?: number;
    showAll?: boolean;
    onToggleShowAll?: () => void;
    rows?: number;
    columns?: number;
}

const MIN_ROWS_FOR_INPUT = 5;
const TRUNCATION_INDICATOR_MARGIN_Y = 1;
const TRUNCATION_INDICATOR_PADDING_X = 1;
const TRUNCATION_INDICATOR_BORDER_HEIGHT = 2;

/**
 * Overall message history container, mirroring OI2 from golden source.
 * Handles windowed rendering (virtualization) based on terminal height.
 */
export const Transcript: React.FC<TranscriptProps> = ({
    messages,
    scrollOffset: _scrollOffset = 0,
    showAll = false,
    onToggleShowAll: _onToggleShowAll,
    rows = 24,
    columns = 80
}) => {
    // Determine how many messages to show based on terminal height
    // We leave some space for the header, input area, and status line
    const availableHeight = Math.max(5, rows - MIN_ROWS_FOR_INPUT - 2);
    const effectiveColumns = Math.max(1, columns - 4);

    const estimateTruncationIndicatorHeight = (text: string) => {
        const contentWidth = Math.max(
            1,
            effectiveColumns - (TRUNCATION_INDICATOR_PADDING_X * 2 + 2)
        );
        const textLines = MeasuredText.measureHeight(text, contentWidth);
        return textLines + TRUNCATION_INDICATOR_BORDER_HEIGHT + TRUNCATION_INDICATOR_MARGIN_Y * 2;
    };

    const { visibleMessages, hasTruncatedMessages, truncatedCount, canTruncate } = useMemo(() => {
        if (showAll) {
            if (messages.length === 0) {
                return { visibleMessages: [], hasTruncatedMessages: false, truncatedCount: 0, canTruncate: false };
            }
            const messageHeights = messages.map(message =>
                Math.max(1, MessageHeightCalculator.calculateHeight(message, effectiveColumns))
            );
            const totalHeight = messageHeights.reduce((sum, height) => sum + height, 0);
            return { visibleMessages: messages, hasTruncatedMessages: false, truncatedCount: 0, canTruncate: totalHeight > availableHeight };
        }

        if (messages.length === 0) {
            return { visibleMessages: [], hasTruncatedMessages: false, truncatedCount: 0, canTruncate: false };
        }

        const messageHeights = messages.map(message =>
            Math.max(1, MessageHeightCalculator.calculateHeight(message, effectiveColumns))
        );
        const totalHeight = messageHeights.reduce((sum, height) => sum + height, 0);
        const canTruncate = totalHeight > availableHeight;

        if (!canTruncate) {
            return { visibleMessages: messages, hasTruncatedMessages: false, truncatedCount: 0, canTruncate: false };
        }

        const indicatorTextFor = (count: number) => `[Ctrl+E] to show ${count} previous messages`;
        const computeStartIndex = (reservedHeight: number) => {
            let remaining = Math.max(0, availableHeight - reservedHeight);
            let startIndex = messages.length;

            for (let i = messages.length - 1; i >= 0; i--) {
                const messageHeight = messageHeights[i];
                if (remaining - messageHeight < 0) {
                    if (startIndex === messages.length) {
                        startIndex = i;
                    }
                    break;
                }
                remaining -= messageHeight;
                startIndex = i;
            }

            return startIndex;
        };

        let startIndex = computeStartIndex(0);
        let truncatedCount = messages.length - startIndex;
        let indicatorHeight = estimateTruncationIndicatorHeight(indicatorTextFor(truncatedCount));

        startIndex = computeStartIndex(indicatorHeight);
        truncatedCount = messages.length - startIndex;

        const updatedIndicatorHeight = estimateTruncationIndicatorHeight(indicatorTextFor(truncatedCount));
        if (updatedIndicatorHeight !== indicatorHeight) {
            indicatorHeight = updatedIndicatorHeight;
            startIndex = computeStartIndex(indicatorHeight);
            truncatedCount = messages.length - startIndex;
        }

        return {
            visibleMessages: messages.slice(startIndex),
            hasTruncatedMessages: startIndex > 0,
            truncatedCount,
            canTruncate: true
        };
    }, [messages, showAll, availableHeight, effectiveColumns]);

    return (
        <Box
            flexDirection="column"
            flexGrow={1}
            overflowY="hidden"
            width="100%"
        >
            <Box flexDirection="column" marginTop={0}>
                {hasTruncatedMessages && (
                    <Box borderStyle="round" borderColor="dim" paddingX={1} marginY={1} justifyContent="center">
                        <Text dimColor italic>
                            [Ctrl+E] to show {truncatedCount} previous messages
                        </Text>
                    </Box>
                )}

                {visibleMessages.map((msg, index) => (
                    <MessageRenderer
                        key={msg.uuid || index}
                        message={msg}
                    />
                ))}

                {showAll && canTruncate && (
                    <Box borderStyle="round" borderColor="dim" paddingX={1} marginY={1} justifyContent="center">
                        <Text dimColor italic>
                            [Ctrl+E] to hide older messages
                        </Text>
                    </Box>
                )}
            </Box>
        </Box>
    );
};
