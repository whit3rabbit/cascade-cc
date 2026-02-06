import React from 'react';
import { Box, Text } from 'ink';
import { markdownRenderer } from '../../utils/text/MarkdownRenderer.js';

export interface MessageRendererProps {
    message: {
        role: string;
        content: any;
        uuid?: string;
    };
    verbose?: boolean;
    isTranscriptMode?: boolean;
}

/**
 * Modular renderer for individual messages, mirroring M62 from golden source.
 */
export const MessageRenderer: React.FC<MessageRendererProps> = ({
    message,
    verbose = false,
    isTranscriptMode: _isTranscriptMode = true
}) => {
    const { role, content } = message;

    const renderTextContent = (text: string) => {
        const tokens = markdownRenderer.parse(text);
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
                default:
                    return (
                        <Text key={index} dimColor>
                            {token.raw || token.text}
                        </Text>
                    );
            }
        });
    };

    const renderContent = (content: any) => {
        if (Array.isArray(content)) {
            return content.map((block, idx) => {
                if (block.type === 'text') {
                    return <Box key={idx} flexDirection="column">{renderTextContent(block.text)}</Box>;
                }
                if (block.type === 'tool_use') {
                    return (
                        <Box key={idx} paddingX={1} borderStyle="round" borderColor="yellow" marginY={1}>
                            <Text bold color="yellow">Tool Use: {block.name}</Text>
                            {verbose && <Text dimColor>{JSON.stringify(block.input, null, 2)}</Text>}
                        </Box>
                    );
                }
                if (block.type === 'tool_result') {
                    // Check for micro-compaction pattern
                    let contentToDisplay = String(block.content);
                    let isMicroCompacted = false;
                    const match = contentToDisplay.match(/Tool result saved to: (.+)\n/);

                    if (match && match[1]) {
                        try {
                            const { readFileSync, existsSync } = require('fs');
                            const savedPath = match[1].trim();
                            if (existsSync(savedPath)) {
                                const fileContent = readFileSync(savedPath, 'utf-8');
                                // Display first 5 lines or similar as preview
                                const lines = fileContent.split('\n');
                                const preview = lines.slice(0, 5).join('\n');
                                contentToDisplay = `[Micro-compacted to ${savedPath}]\n\nPreview:\n${preview}${lines.length > 5 ? `\n... (${lines.length - 5} more lines)` : ''}`;
                                isMicroCompacted = true;
                            }
                        } catch {
                            // Ignore read errors, show original message
                        }
                    }

                    return (
                        <Box key={idx} paddingX={1} borderStyle="round" borderColor="green" marginY={1} flexDirection="column">
                            <Text bold color="green">Tool Result: {block.tool_use_id}</Text>
                            <Box marginTop={0}>
                                <Text dimColor={!isMicroCompacted}>{contentToDisplay}</Text>
                            </Box>
                        </Box>
                    );
                }
                return null;
            });
        }
        return <Box flexDirection="column">{renderTextContent(String(content))}</Box>;
    };

    return (
        <Box flexDirection="column" marginY={1} width="100%">
            <Box>
                <Text bold color={role === 'user' ? 'white' : 'blue'}>
                    {role === 'user' ? 'User' : 'Claude'}
                </Text>
            </Box>
            <Box flexDirection="column" paddingLeft={2}>
                {renderContent(content)}
            </Box>
        </Box>
    );
};
