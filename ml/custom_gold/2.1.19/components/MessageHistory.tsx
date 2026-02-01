/**
 * File: src/components/MessageHistory.tsx
 * Role: Renders the history of messages in the conversation.
 */

import React, { useMemo } from 'react';
import { Box, Text, useStdout } from 'ink';
import { highlight } from 'cli-highlight';
import chalk from 'chalk';

export interface MessageHistoryProps {
    messages: any[];
    scrollOffset?: number;
    terminalHeight?: number; // Optional override
}


import { markdownRenderer } from '../utils/text/MarkdownRenderer.js';

const renderContent = (content: any) => {
    if (!content) return null;

    // Handle Anthropic-style content arrays
    let textContent = '';
    if (Array.isArray(content)) {
        textContent = content
            .map(block => {
                if (block.type === 'text') return block.text;
                if (block.type === 'tool_use') return `[Tool Use: ${block.name}]`;
                return '';
            })
            .join('\n');
    } else {
        textContent = String(content);
    }

    const tokens = markdownRenderer.parse(textContent);
    return tokens.map((token: any, index: number) => {
        switch (token.type) {
            case 'code':
                const lang = token.lang || 'plaintext';
                const highlighted = markdownRenderer.highlightCode(token.text, lang);
                return (
                    <Box key={index} borderStyle="single" borderColor="gray" paddingX={1} flexDirection="column" marginY={1}>
                        {lang && <Text dimColor>{lang}</Text>}
                        <Text>{highlighted}</Text>
                    </Box>
                );
            case 'paragraph':
                return (
                    <Text key={index}>
                        {markdownRenderer.formatInline(token.text)}
                    </Text>
                );
            case 'heading':
                return (
                    <Box key={index} marginTop={1}>
                        <Text bold color="yellow" underline>
                            {'#'.repeat(token.depth)} {markdownRenderer.formatInline(token.text)}
                        </Text>
                    </Box>
                );
            case 'list':
                return (
                    <Box key={index} flexDirection="column" paddingLeft={2} marginY={1}>
                        <Text>{markdownRenderer.renderBlockAsString(token)}</Text>
                    </Box>
                );
            case 'blockquote':
                return (
                    <Box key={index} borderStyle="single" borderColor="blue" paddingX={1} marginY={1}>
                        <Text italic dimColor>
                            {markdownRenderer.formatInline(token.text)}
                        </Text>
                    </Box>
                );
            case 'space':
                return null;
            default:
                return (
                    <Text key={index} dimColor>
                        {token.raw || token.text}
                    </Text>
                );
        }
    });
};

export const MessageHistory: React.FC<MessageHistoryProps> = ({ messages, scrollOffset = 0, terminalHeight: terminalHeightOverride }) => {
    const { stdout } = useStdout();
    const rows = terminalHeightOverride || stdout?.rows || 24;

    return (
        <Box
            flexDirection="column"
            height={rows - 5}
            overflowY="hidden"
        >
            <Box flexDirection="column" gap={1} marginTop={-scrollOffset}>
                {messages.map((msg, index) => (
                    <Box key={index} flexDirection="column" borderStyle={msg.role === 'assistant' ? 'round' : undefined} borderColor={msg.role === 'assistant' ? "blue" : "gray"} padding={msg.role === 'assistant' ? 1 : 0}>
                        <Text bold color={msg.role === 'user' ? 'white' : 'blue'}>
                            {msg.role === 'user' ? 'User' : 'Claude'}
                        </Text>
                        <Box flexDirection="column">
                            {renderContent(msg.content)}
                        </Box>
                    </Box>
                ))}
            </Box>
        </Box>
    );
};
