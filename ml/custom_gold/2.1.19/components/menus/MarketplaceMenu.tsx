
import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { MarketplaceLoader } from '../../services/marketplace/MarketplaceLoader.js';
import { PluginManager, PluginInstallation } from '../../services/mcp/PluginManager.js';
import Spinner from 'ink-spinner';

interface MarketplaceMenuProps {
    onExit: () => void;
}

function getDirectoryPluginPrimaryKey(plugin: any): string | null {
    return plugin?.id || plugin?.pluginId || plugin?.repository || plugin?.name || null;
}

function getDirectoryPluginKeys(plugin: any): string[] {
    const keys = new Set<string>();
    if (plugin?.id) keys.add(plugin.id);
    if (plugin?.pluginId) keys.add(plugin.pluginId);
    if (plugin?.name) keys.add(plugin.name);
    if (plugin?.repository) {
        keys.add(plugin.repository);
        const repoName = String(plugin.repository).split('/').pop();
        if (repoName) keys.add(repoName);
    }
    return Array.from(keys);
}

function getInstalledPluginKeys(plugin: PluginInstallation): string[] {
    const keys = new Set<string>();
    if (plugin.id) {
        keys.add(plugin.id);
        if (plugin.id.startsWith('plugin:')) {
            keys.add(plugin.id.replace('plugin:', ''));
        }
        if (plugin.id.includes('@')) {
            keys.add(plugin.id.split('@')[0]);
        }
    }
    if (plugin.name) keys.add(plugin.name);
    const repository = (plugin as any).repository;
    if (repository) {
        keys.add(repository);
        const repoName = String(repository).split('/').pop();
        if (repoName) keys.add(repoName);
    }
    return Array.from(keys);
}

function buildInstalledPluginKeySet(installed: PluginInstallation[]): Set<string> {
    const installedSet = new Set<string>();
    for (const plugin of installed) {
        for (const key of getInstalledPluginKeys(plugin)) {
            installedSet.add(key);
        }
    }
    return installedSet;
}

function isDirectoryPluginInstalled(plugin: any, installedKeys: Set<string>): boolean {
    const keys = getDirectoryPluginKeys(plugin);
    return keys.some(key => installedKeys.has(key));
}

function resolveInstalledPluginId(plugin: any): string | null {
    if (plugin?.id && String(plugin.id).startsWith('plugin:')) {
        return plugin.id;
    }
    const repository = plugin?.repository;
    if (repository) {
        const repoName = String(repository).split('/').pop();
        if (repoName) return `plugin:${repoName}`;
    }
    if (plugin?.name) {
        return `plugin:${plugin.name}`;
    }
    return null;
}

function dedupePlugins(directory: any[]): any[] {
    const seen = new Set<string>();
    const deduped: any[] = [];
    for (const plugin of directory) {
        const key = getDirectoryPluginPrimaryKey(plugin);
        if (key && seen.has(key)) continue;
        if (key) seen.add(key);
        deduped.push(plugin);
    }
    return deduped;
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

            const deduped = dedupePlugins(directory);
            setPlugins(deduped);
            setInstalledPlugins(buildInstalledPluginKeySet(installed));
        } catch (e) {
            setError(`Failed to load marketplace: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleTogglePlugin = async (plugin: any) => {
        if (installing) return;

        const isInstalled = isDirectoryPluginInstalled(plugin, installedPlugins);
        setInstalling(plugin.name);
        setStatusMessage(isInstalled ? "Uninstalling..." : "Installing...");

        try {
            if (isInstalled) {
                const pluginId = resolveInstalledPluginId(plugin);
                if (!pluginId) {
                    setError(`Unable to determine installed plugin ID for ${plugin.name}`);
                } else {
                    const result = await PluginManager.uninstallPlugin(pluginId, 'user');
                    if (result.success) {
                        setStatusMessage(`Uninstalled ${plugin.name}`);
                        const newInstalled = new Set(installedPlugins);
                        for (const key of getDirectoryPluginKeys(plugin)) {
                            newInstalled.delete(key);
                        }
                        setInstalledPlugins(newInstalled);
                    } else {
                        setError(result.message);
                    }
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
                    for (const key of getDirectoryPluginKeys(plugin)) {
                        newInstalled.add(key);
                    }
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
                    const isInstalled = isDirectoryPluginInstalled(plugin, installedPlugins);
                    const isProcessing = installing === plugin.name;

                    return (
                        <Box key={getDirectoryPluginPrimaryKey(plugin) || i} flexDirection="row">
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
