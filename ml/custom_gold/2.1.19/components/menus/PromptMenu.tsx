
import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { mcpClientManager } from '../../services/mcp/McpClientManager.js';
import Spinner from 'ink-spinner';

interface PromptMenuProps {
    onExit: () => void;
}

export function PromptMenu({ onExit }: PromptMenuProps) {
    const [loading, setLoading] = useState(true);
    const [prompts, setPrompts] = useState<any[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadPrompts();
    }, []);

    const loadPrompts = async () => {
        setLoading(true);
        try {
            const res = await mcpClientManager.getPrompts();
            setPrompts(res);
        } catch (e) {
            setError(`Failed to load prompts: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    useInput((input, key) => {
        if (key.escape) {
            onExit();
            return;
        }

        if (loading) return;

        if (key.upArrow) {
            setSelectedIndex(prev => Math.max(0, prev - 1));
        }
        if (key.downArrow) {
            setSelectedIndex(prev => Math.min(prompts.length - 1, prev + 1));
        }

        if (key.return) {
            // Placeholder for execution or view
        }
    });

    if (loading) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>
                    <Spinner type="dots" /> Loading prompts...
                </Text>
            </Box>
        );
    }

    if (error) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text color="red">Error: {error}</Text>
                <Text dimColor>Press Esc to go back</Text>
            </Box>
        );
    }

    if (prompts.length === 0) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>No prompts found from active MCP servers.</Text>
                <Text dimColor>Press Esc to go back</Text>
            </Box>
        );
    }

    const flatList = prompts;

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} width={80}>
            <Text bold>MCP Prompts</Text>
            <Text dimColor>{prompts.length} prompts available</Text>

            <Box marginTop={1} flexDirection="column">
                {flatList.map((p, i) => {
                    const isSelected = i === selectedIndex;
                    return (
                        <Box key={p.name + i} flexDirection="row">
                            <Text color={isSelected ? 'cyan' : 'white'}>
                                {isSelected ? '❯ ' : '  '}
                            </Text>
                            <Text>
                                {p.name} <Text dimColor>{p.description ? `- ${p.description}` : ''}</Text>
                            </Text>
                        </Box>
                    );
                })}
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Press Enter to use (TODO) · Esc to go back</Text>
            </Box>
        </Box>
    );
}
