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

type MarketplacePlugin = {
    pluginId: string;
    name?: string;
    version?: string;
    description?: string;
    author?: string | { name?: string };
    repository?: string;
    marketplace?: string;
    [key: string]: any;
};

export function McpDiscoverView() {
    const [plugins, setPlugins] = useState<MarketplacePlugin[]>([]);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [focusedIndex, setFocusedIndex] = useState(0);
    const [installing, setInstalling] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [detailPlugin, setDetailPlugin] = useState<MarketplacePlugin | null>(null);

    useEffect(() => {
        const fetchMarketplaces = async () => {
            const marketplaces = await MarketplaceService.listMarketplaces();
            const allPlugins: any[] = [];
            marketplaces.forEach(m => {
                if (m.config.plugins) {
                    allPlugins.push(...m.config.plugins.map((p: any) => ({
                        ...p,
                        marketplace: m.name,
                        pluginId: buildMarketplacePluginId(p, m.name)
                    })));
                }
            });

            if (allPlugins.length === 0) {
                // If nothing found, try to refresh all
                await MarketplaceService.refreshAllMarketplaces();
                const refreshed = await MarketplaceService.listMarketplaces();
                refreshed.forEach(m => {
                    if (m.config.plugins) {
                        allPlugins.push(...m.config.plugins.map((p: any) => ({
                            ...p,
                            marketplace: m.name,
                            pluginId: buildMarketplacePluginId(p, m.name)
                        })));
                    }
                });
            }
            setPlugins(dedupePlugins(allPlugins));
        };
        fetchMarketplaces();
    }, []);

    useInput(async (input, _key) => {
        if (installing) return;
        if (input === ' ') {
            const plugin = plugins[focusedIndex];
            if (plugin) {
                const newSelection = new Set(selectedIds);
                if (newSelection.has(plugin.pluginId)) {
                    newSelection.delete(plugin.pluginId);
                } else {
                    newSelection.add(plugin.pluginId);
                }
                setSelectedIds(newSelection);
            }
        }
        if (input === 'i' && selectedIds.size > 0) {
            setInstalling(true);
            const idsToInstall = Array.from(selectedIds);
            for (let i = 0; i < idsToInstall.length; i++) {
                const id = idsToInstall[i];
                const plugin = plugins.find(p => p.pluginId === id);
                if (plugin) {
                    setStatusMessage(`Installing ${plugin.name} (${i + 1}/${idsToInstall.length})...`);
                    await installPlugin(plugin, "user");
                }
            }
            setStatusMessage("Installation complete!");
            setTimeout(() => setStatusMessage(null), 3000);
            setSelectedIds(new Set());
            setDetailPlugin(null);
            setInstalling(false);
        }
    });

    const handleSelect = (item: any) => {
        const plugin = plugins.find(p => p.pluginId === item.value);
        if (!plugin) return;
        setDetailPlugin((current: MarketplacePlugin | null) => {
            if (current?.pluginId === plugin.pluginId) return null;
            return plugin;
        });
    };

    const handleHighlight = (item: any) => {
        const index = plugins.findIndex(p => p.pluginId === item.value);
        if (index !== -1) setFocusedIndex(index);
    };

    const items = plugins.map(p => ({
        label: `${selectedIds.has(p.pluginId) ? '[X] ' : '[ ] '} ${p.name}`,
        value: p.pluginId
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
                    {detailPlugin && (
                        <Box marginTop={1} paddingX={1} borderStyle="round" borderColor="gray">
                            <Box flexDirection="column">
                                <Text bold>{detailPlugin.name}</Text>
                                {detailPlugin.version && <Text dimColor>Version: {detailPlugin.version}</Text>}
                                {detailPlugin.description && <Text>{detailPlugin.description}</Text>}
                                {detailPlugin.author && (
                                    <Text dimColor>By: {typeof detailPlugin.author === 'string' ? detailPlugin.author : detailPlugin.author.name}</Text>
                                )}
                                {detailPlugin.repository && <Text dimColor>Repo: {detailPlugin.repository}</Text>}
                                {detailPlugin.marketplace && <Text dimColor>Marketplace: {detailPlugin.marketplace}</Text>}
                            </Box>
                        </Box>
                    )}
                </Box>
            ) : (
                <Text dimColor>Loading marketplaces...</Text>
            )}
        </Box>
    );
}

function buildMarketplacePluginId(plugin: any, marketplaceName: string): string {
    const name = plugin?.name || plugin?.repository || "unknown";
    return `${name}@${marketplaceName}`;
}

function dedupePlugins(entries: any[]): any[] {
    const seen = new Set<string>();
    const output: any[] = [];
    for (const entry of entries) {
        const key = entry?.pluginId || entry?.id || entry?.repository || entry?.name;
        if (key && seen.has(key)) continue;
        if (key) seen.add(key);
        output.push(entry);
    }
    return output;
}
