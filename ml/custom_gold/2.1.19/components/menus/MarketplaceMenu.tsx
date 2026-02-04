
import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { MarketplaceLoader } from '../../services/mcp/MarketplaceLoader.js';
import { PluginManager } from '../../services/mcp/PluginManager.js';
import Spinner from 'ink-spinner';

interface MarketplaceMenuProps {
    onExit: () => void;
}

export function MarketplaceMenu({ onExit }: MarketplaceMenuProps) {
    const [loading, setLoading] = useState(true);
    const [plugins, setPlugins] = useState<any[]>([]);
    const [installedPlugins, setInstalledPlugins] = useState<Set<string>>(new Set());
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [installing, setInstalling] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);

    // Initial load
    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        setLoading(true);
        try {
            const [directory, installed] = await Promise.all([
                MarketplaceLoader.fetchPluginDirectory(),
                PluginManager.getInstalledPlugins()
            ]);

            setPlugins(directory);

            const installedSet = new Set<string>();
            installed.forEach(p => {
                // Determine ID from name or repo
                // The directory usually has 'name' or 'repository'.
                // PluginManager uses 'plugin:name' or 'plugin:repo'.
                // We'll try to match by name or repository URL if available.
                installedSet.add(p.name);
                if (p.id) installedSet.add(p.id.replace('plugin:', ''));
            });
            setInstalledPlugins(installedSet);
        } catch (e) {
            setError(`Failed to load marketplace: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleTogglePlugin = async (plugin: any) => {
        if (installing) return;

        const isInstalled = installedPlugins.has(plugin.name);
        setInstalling(plugin.name);
        setStatusMessage(isInstalled ? "Uninstalling..." : "Installing...");

        try {
            if (isInstalled) {
                const result = await PluginManager.uninstallPlugin(`plugin:${plugin.name}`, 'user');
                if (result.success) {
                    setStatusMessage(`Uninstalled ${plugin.name}`);
                    const newInstalled = new Set(installedPlugins);
                    newInstalled.delete(plugin.name);
                    setInstalledPlugins(newInstalled);
                } else {
                    setError(result.message);
                }
            } else {
                // Install
                const result = await PluginManager.installPlugin({
                    source: plugin.source || 'github', // default to github
                    repository: plugin.repository,
                    name: plugin.name,
                    version: plugin.version,
                    mcp: plugin.mcp
                }, 'user');

                if (result.success) {
                    setStatusMessage(`Installed ${plugin.name}`);
                    const newInstalled = new Set(installedPlugins);
                    newInstalled.add(plugin.name);
                    setInstalledPlugins(newInstalled);
                } else {
                    setError(result.message);
                }
            }
        } catch (e) {
            setError(`Operation failed: ${(e as Error).message}`);
        } finally {
            setInstalling(null);
            // Clear status message after 3 seconds
            setTimeout(() => setStatusMessage(null), 3000);
        }
    };

    useInput((input, key) => {
        if (key.escape) {
            onExit();
            return;
        }

        if (loading || installing) return;

        if (key.upArrow) {
            setSelectedIndex(prev => Math.max(0, prev - 1));
        }
        if (key.downArrow) {
            setSelectedIndex(prev => Math.min(plugins.length - 1, prev + 1));
        }

        if (key.return || input === ' ') {
            const plugin = plugins[selectedIndex];
            if (plugin) {
                handleTogglePlugin(plugin);
            }
        }
    });

    if (loading) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>
                    <Spinner type="dots" /> Loading marketplace...
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

    if (plugins.length === 0) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>No plugins found in the marketplace.</Text>
                <Text dimColor>Press Esc to go back</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} width={80}>
            <Text bold>MCP Marketplace</Text>
            <Text dimColor>Official Plugins</Text>

            <Box marginTop={1} flexDirection="column">
                {plugins.map((plugin, i) => {
                    const isSelected = i === selectedIndex;
                    const isInstalled = installedPlugins.has(plugin.name);
                    const isProcessing = installing === plugin.name;

                    return (
                        <Box key={plugin.name || i} flexDirection="row">
                            <Text color={isSelected ? 'cyan' : 'white'}>
                                {isSelected ? '❯ ' : '  '}
                            </Text>
                            <Text color={isInstalled ? 'green' : 'white'}>
                                {isInstalled ? '[Installed] ' : '[         ] '}
                            </Text>
                            <Text bold={isSelected}>
                                {plugin.name}
                            </Text>
                            <Text dimColor>
                                {plugin.description ? ` - ${plugin.description}` : ''}
                            </Text>
                            {isProcessing && <Text color="yellow"> ...</Text>}
                        </Box>
                    );
                })}
            </Box>

            <Box marginTop={1} flexDirection="column">
                {statusMessage && <Text color="green">{statusMessage}</Text>}
                <Text dimColor>Press Enter/Space to install/uninstall · Esc to go back</Text>
            </Box>
        </Box>
    );
}
