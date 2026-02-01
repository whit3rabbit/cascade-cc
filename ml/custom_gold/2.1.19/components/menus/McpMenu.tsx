/**
 * File: src/components/menus/McpMenu.tsx
 * Role: Interface for managing Model Context Protocol (MCP) servers.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Box, Text, useInput } from 'ink';
import { mcpClientManager } from '../../services/mcp/McpClientManager.js';
import { McpServerManager } from '../../services/mcp/McpServerManager.js';
import os from 'os';

interface McpMenuProps {
    onExit: () => void;
}

interface ServerItem {
    id: string;
    type: 'user' | 'builtin';
    status: 'connected' | 'disabled' | 'error';
    label: string;
}

export function McpMenu({ onExit }: McpMenuProps) {
    const [activeClients, setActiveClients] = useState<string[]>([]);
    const [configuredServers, setConfiguredServers] = useState<any>({});
    const [selectedIndex, setSelectedIndex] = useState(0);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        const active = mcpClientManager.getActiveClients();
        setActiveClients(active);

        try {
            const configured = await McpServerManager.getAllMcpServers();
            setConfiguredServers(configured);
        } catch (e) {
            // Fallback if manager fails
            setConfiguredServers({});
        }
    };

    const items: ServerItem[] = useMemo(() => {
        const list: ServerItem[] = [];

        // 1. User MCPs (from config settings)
        Object.keys(configuredServers).forEach(name => {
            const isConnected = activeClients.includes(name);
            // If it's not a plugin, we assume it's a user MCP
            if (!name.startsWith('plugin:')) {
                list.push({
                    id: name,
                    type: 'user',
                    status: isConnected ? 'connected' : 'disabled',
                    label: name
                });
            }
        });

        // Add user MCPs that might be active but not in config (dynamic/adhoc)
        activeClients.forEach(name => {
            if (!name.startsWith('plugin:') && !configuredServers[name]) {
                list.push({
                    id: name,
                    type: 'user',
                    status: 'connected',
                    label: name
                });
            }
        });

        // 2. Built-in MCPs
        activeClients.forEach(name => {
            if (name.startsWith('plugin:')) {
                list.push({
                    id: name,
                    type: 'builtin',
                    status: 'connected',
                    label: name
                });
            }
        });

        // Deduplicate by ID just in case
        return Array.from(new Map(list.map(item => [item.id, item])).values());
    }, [activeClients, configuredServers]);

    const userMcpItems = items.filter(i => i.type === 'user');
    const builtinItems = items.filter(i => i.type === 'builtin');

    // Flatten for navigation
    const flatItems = [...userMcpItems, ...builtinItems];

    useInput((input, key) => {
        if (key.escape) {
            onExit();
        }
        if (key.upArrow) {
            setSelectedIndex(prev => Math.max(0, prev - 1));
        }
        if (key.downArrow) {
            setSelectedIndex(prev => Math.min(flatItems.length - 1, prev + 1));
        }
        if (key.return) {
            // Toggle functionality could go here (connect/disconnect)
            // For now we just acknowledge or maybe toggle if implemented
            const item = flatItems[selectedIndex];
            if (item) {
                // handleToggle(item);
            }
        }
    });

    const renderItem = (item: ServerItem, globalIndex: number) => {
        const isSelected = globalIndex === selectedIndex;
        const icon = item.status === 'connected' ? '✔' : '◯';
        const statusText = item.status;

        return (
            <Box key={item.id} marginLeft={2}>
                <Text color={isSelected ? 'cyan' : 'white'}>
                    {isSelected ? '❯ ' : '  '}
                    {item.label} · <Text color={item.status === 'connected' ? 'green' : 'gray'}>{icon} {statusText}</Text>
                </Text>
            </Box>
        );
    };

    let currentIndex = 0;

    return (
        <Box flexDirection="column" paddingX={1} width={80}>
            <Text bold>Manage MCP servers</Text>
            <Text dimColor>{items.length} servers</Text>

            <Box marginTop={1} flexDirection="column">
                <Text dimColor>User MCPs ({os.homedir()}/.claude.json)</Text>
                {userMcpItems.length > 0 ? (
                    userMcpItems.map(item => renderItem(item, currentIndex++))
                ) : (
                    <Box marginLeft={2}><Text dimColor>No user servers configured</Text></Box>
                )}
            </Box>

            <Box marginTop={1} flexDirection="column">
                <Text dimColor>Built-in MCPs (always available)</Text>
                {builtinItems.length > 0 ? (
                    builtinItems.map(item => renderItem(item, currentIndex++))
                ) : (
                    <Box marginLeft={2}><Text dimColor>No built-in servers active</Text></Box>
                )}
            </Box>

            <Box marginTop={1} flexDirection="column">
                <Text color="blue">https://code.claude.com/docs/en/mcp for help</Text>
            </Box>

            <Box marginTop={1}>
                <Text dimColor>↑↓ to navigate · Enter to confirm · Esc to cancel</Text>
            </Box>
        </Box>
    );
}
