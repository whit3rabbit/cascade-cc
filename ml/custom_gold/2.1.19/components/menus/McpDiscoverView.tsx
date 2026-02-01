/**
 * File: src/components/menus/McpDiscoverView.tsx
 * Role: UI for discovering and installing MCP servers from marketplaces.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { MarketplaceService } from '../../services/marketplace/MarketplaceService.js';
import { MCPServerMultiselectDialog } from '../mcp/MCPServerDialog.js';
import { installPlugin } from '../../services/mcp/PluginManager.js';

export function McpDiscoverView() {
    const [plugins, setPlugins] = useState<any[]>([]);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [focusedIndex, setFocusedIndex] = useState(0);
    const [installing, setInstalling] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);

    useEffect(() => {
        const fetchMarketplaces = async () => {
            const marketplaces = await MarketplaceService.listMarketplaces();
            const allPlugins: any[] = [];
            marketplaces.forEach(m => {
                if (m.config.plugins) {
                    allPlugins.push(...m.config.plugins.map((p: any) => ({ ...p, marketplace: m.name })));
                }
            });

            if (allPlugins.length === 0) {
                // If nothing found, try to refresh all
                await MarketplaceService.refreshAllMarketplaces();
                const refreshed = await MarketplaceService.listMarketplaces();
                refreshed.forEach(m => {
                    if (m.config.plugins) {
                        allPlugins.push(...m.config.plugins.map((p: any) => ({ ...p, marketplace: m.name })));
                    }
                });
            }
            setPlugins(allPlugins);
        };
        fetchMarketplaces();
    }, []);

    useInput(async (input, key) => {
        if (installing) return;
        if (input === ' ') {
            const plugin = plugins[focusedIndex];
            if (plugin) {
                const newSelection = new Set(selectedIds);
                if (newSelection.has(plugin.id)) {
                    newSelection.delete(plugin.id);
                } else {
                    newSelection.add(plugin.id);
                }
                setSelectedIds(newSelection);
            }
        }
        if (input === 'i' && selectedIds.size > 0) {
            setInstalling(true);
            const idsToInstall = Array.from(selectedIds);
            for (let i = 0; i < idsToInstall.length; i++) {
                const id = idsToInstall[i];
                const plugin = plugins.find(p => p.id === id);
                if (plugin) {
                    setStatusMessage(`Installing ${plugin.name} (${i + 1}/${idsToInstall.length})...`);
                    await installPlugin(plugin, "user");
                }
            }
            setStatusMessage("Installation complete!");
            setTimeout(() => setStatusMessage(null), 3000);
            setSelectedIds(new Set());
            setInstalling(false);
        }
    });

    const handleSelect = (item: any) => {
        // Maybe show details on enter
    };

    const handleHighlight = (item: any) => {
        const index = plugins.findIndex(p => p.id === item.value);
        if (index !== -1) setFocusedIndex(index);
    };

    const items = plugins.map(p => ({
        label: `${selectedIds.has(p.id) ? '[X] ' : '[ ] '} ${p.name}`,
        value: p.id
    }));

    return (
        <Box flexDirection="column">
            <Text bold color="cyan">Discover MCP Servers</Text>
            {statusMessage && (
                <Box marginTop={1} paddingX={1} borderStyle="single" borderColor="yellow">
                    <Text color="yellow">{statusMessage}</Text>
                </Box>
            )}
            {plugins.length > 0 ? (
                <Box flexDirection="column" marginTop={1}>
                    {!installing ? (
                        <SelectInput
                            items={items}
                            onSelect={handleSelect}
                            onHighlight={handleHighlight}
                        />
                    ) : (
                        <Text dimColor>Installation in progress...</Text>
                    )}
                    <MCPServerMultiselectDialog hasSelection={selectedIds.size > 0} />
                </Box>
            ) : (
                <Text dimColor>Loading marketplaces...</Text>
            )}
        </Box>
    );
}
