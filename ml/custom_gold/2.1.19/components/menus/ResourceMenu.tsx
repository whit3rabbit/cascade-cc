import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { mcpClientManager } from '../../services/mcp/McpClientManager.js';
import Spinner from 'ink-spinner';

interface ResourceMenuProps {
    onExit: () => void;
}

export function ResourceMenu({ onExit }: ResourceMenuProps) {
    const [loading, setLoading] = useState(true);
    const [resources, setResources] = useState<any[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [content, setContent] = useState<string | null>(null);
    const [viewingResource, setViewingResource] = useState<string | null>(null);

    useEffect(() => {
        loadResources();
    }, []);

    const loadResources = async () => {
        setLoading(true);
        try {
            const res = await mcpClientManager.getResources();
            setResources(res);
        } catch (e) {
            setError(`Failed to load resources: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    const flatList = resources;

    useInput(async (input, key) => {
        if (key.escape) {
            if (content) {
                setContent(null);
                setViewingResource(null);
                return;
            }
            onExit();
            return;
        }

        if (loading) return;

        if (content) {
            return;
        }

        if (key.upArrow) {
            setSelectedIndex(prev => Math.max(0, prev - 1));
        }
        if (key.downArrow) {
            setSelectedIndex(prev => Math.min(flatList.length - 1, prev + 1));
        }

        if (key.return) {
            const resource = flatList[selectedIndex];
            if (resource) {
                setLoading(true);
                try {
                    const res = await mcpClientManager.readResource(resource.serverId, resource.uri);
                    const actualContent = res.contents?.[0];
                    let textToShow = "No content";
                    if (actualContent) {
                        textToShow = actualContent.text || (actualContent.blob ? "[Binary Data]" : "Empty");
                    }
                    setContent(textToShow);
                    setViewingResource(resource.name);
                } catch (e) {
                    setError(`Failed to read resource: ${(e as Error).message}`);
                } finally {
                    setLoading(false);
                }
            }
        }
    });

    if (loading) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>
                    <Spinner type="dots" /> {content ? 'Reading resource...' : 'Loading resources...'}
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

    if (content) {
        return (
            <Box flexDirection="column" borderStyle="round" borderColor="cyan" padding={1} width={80}>
                <Text bold>Resource: {viewingResource}</Text>
                <Box marginY={1} borderStyle="single" borderColor="gray" padding={1}>
                    <Text>{content.slice(0, 2000) + (content.length > 2000 ? '\n... (truncated)' : '')}</Text>
                </Box>
                <Text dimColor>Press Esc to back</Text>
            </Box>
        );
    }

    if (resources.length === 0) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>No resources found from active MCP servers.</Text>
                <Text dimColor>Press Esc to go back</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} width={80}>
            <Text bold>MCP Resources</Text>
            <Text dimColor>{resources.length} resources available</Text>

            <Box marginTop={1} flexDirection="column">
                {flatList.map((r, i) => {
                    const isSelected = i === selectedIndex;
                    return (
                        <Box key={r.uri + i} flexDirection="row">
                            <Text color={isSelected ? 'cyan' : 'white'}>
                                {isSelected ? '❯ ' : '  '}
                            </Text>
                            <Text>
                                {r.name} <Text dimColor>({r.mimeType})</Text>
                            </Text>
                        </Box>
                    );
                })}
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Press Enter to view · Esc to go back</Text>
            </Box>
        </Box>
    );
}
