/**
 * File: src/components/MessageHistory.tsx
 * Role: Renders the history of messages in the conversation.
 */

import React from 'react';
import { Box, Text } from 'ink';
import { highlight } from 'cli-highlight';
import chalk from 'chalk';

export interface MessageHistoryProps {
    messages: any[];
}

import { markdownRenderer } from '../utils/text/MarkdownRenderer.js';

const renderContent = (content: string) => {
    const tokens = markdownRenderer.parse(content);
    return tokens.map((token: any, index: number) => {
        if (token.type === 'code') {
            const lang = token.lang || 'plaintext';
            const highlighted = markdownRenderer.highlightCode(token.text, lang);
            return (
                <Box key={index} borderStyle="single" borderColor="gray" paddingX={1} flexDirection="column" marginY={1}>
                    {lang && <Text dimColor>{lang}</Text>}
                    <Text>{highlighted}</Text>
                </Box>
            );
        } else if (token.type === 'paragraph') {
            return (
                <Text key={index}>
                    {markdownRenderer.formatInline(token.text)}
                </Text>
            );
        } else if (token.type === 'space') {
            return null;
        }

        // Fallback for other token types
        return <Text key={index}>{token.raw}</Text>;
    });
};

export const MessageHistory: React.FC<MessageHistoryProps> = ({ messages }) => {
    return (
        <Box flexDirection="column" gap={1}>
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
    );
};
