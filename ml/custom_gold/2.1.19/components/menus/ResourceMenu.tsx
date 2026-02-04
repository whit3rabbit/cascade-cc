
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
            setSelectedIndex(prev => Math.min(resources.length - 1, prev + 1));
        }

        if (key.return) {
            // TODO: Ideally we would read the resource here or show details
            // For now, maybe just copy the URI to clipboard or Log it?
            // Or render a detail view.
            // Let's just log it to console or show a message for now.
        }
    });

    if (loading) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>
                    <Spinner type="dots" /> Loading resources...
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

    if (resources.length === 0) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>No resources found from active MCP servers.</Text>
                <Text dimColor>Press Esc to go back</Text>
            </Box>
        );
    }

    // Group by server
    const grouped: Record<string, any[]> = {};
    resources.forEach(r => {
        const server = r.serverId || 'unknown';
        if (!grouped[server]) grouped[server] = [];
        grouped[server].push(r);
    });

    // Flatten for selection
    const flatItems: any[] = [];
    Object.keys(grouped).forEach(server => {
        flatItems.push({ type: 'header', label: server });
        grouped[server].forEach(r => {
            flatItems.push({ type: 'item', ...r, serverId: server });
        });
    });

    // Adjust selection if it lands on a header (simple skip logic)
    // Actually, let's just make headers non-selectable in rendering or logic
    // For simplicity, we just list flattened items and skip headers in navigation logic if possible
    // But hooks are hard to sync. Let's just render a flat list for now.

    const renderList = () => {
        let currentIndex = 0;
        return Object.keys(grouped).map(server => (
            <Box key={server} flexDirection="column" marginTop={1}>
                <Text dimColor>Server: {server}</Text>
                {grouped[server].map((r, i) => {
                    const globalIndex = currentIndex++; // This logic is flawed if we want global navigation across groups
                    // A better way is to use a single flat list for index calculation
                    return null;
                })}
            </Box>
        ));
    };

    // Re-do flattening for correct index
    const flatList = resources;

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} width={80}>
            <Text bold>MCP Resources</Text>
            <Text dimColor>{resources.length} resources available</Text>

            <Box marginTop={1} flexDirection="column">
                {flatList.map((r, i) => {
                    const isSelected = i === selectedIndex;
                    return (
                        <Box key={r.uri} flexDirection="row">
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
                <Text dimColor>Press Enter to view (TODO) · Esc to go back</Text>
            </Box>
        </Box>
    );
}
