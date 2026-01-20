
import React, { useState, useEffect, useMemo } from "react";
import { Box, Text, useInput } from "ink";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { figures } from "../../vendor/terminalFigures.js";

import {
    fetchMarketplaces,
    processMarketplaceData,
    fetchPluginInstallCounts,
    handlePluginAction,
    getHomepageActions,
    getGithubUrl
} from "../../services/mcp/PluginManager.js";
import { useScroll } from "../../hooks/useScroll.js";


import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";
import { useTerminalFocus } from "../../hooks/useTerminalFocus.js";

// --- Discovery Footer (xG7) ---
export function DiscoveryFooter({ hasSelection }: { hasSelection: boolean }) {
    return (
        <Box marginLeft={3}>
            <Text italic>
                {hasSelection && (
                    <Text bold color="suggestion">Press i to install · </Text>
                )}
                <Text dimColor>Type to search · Space: (de)select · Enter: details · Esc: back</Text>
            </Text>
        </Box>
    );
}

// --- Empty/Error States (yG7) ---
export function MarketplaceStatusView({ reason }: { reason: string | null }) {
    switch (reason) {
        case "git-not-installed":
            return (
                <Box flexDirection="column">
                    <Text dimColor>Git is required to install marketplaces.</Text>
                    <Text dimColor>Please install git and restart Claude Code.</Text>
                </Box>
            );
        case "all-blocked-by-policy":
            return (
                <Box flexDirection="column">
                    <Text dimColor>Your organization policy does not allow any external marketplaces.</Text>
                    <Text dimColor>Contact your administrator.</Text>
                </Box>
            );
        case "all-plugins-installed":
            return (
                <Text dimColor>All available plugins are already installed.</Text>
            );
        case "no-marketplaces-configured":
            return (
                <Box flexDirection="column">
                    <Text dimColor>No plugins available.</Text>
                    <Text dimColor>Add a marketplace first using the Marketplaces tab.</Text>
                </Box>
            );
        default:
            return (
                <Text dimColor>No plugins found.</Text>
            );
    }
}

/**
 * Main Plugin Discovery Listing.
 * Derived from MX9 in chunk_570.ts.
 */
export function PluginDiscoveryView({
    setError,
    setViewState,
    onInstallComplete,
    targetPlugin,
    children
}: {
    setError: (message: string) => void;
    setViewState: (state: any) => void;
    onInstallComplete: () => void;
    targetPlugin?: string;
    children?: React.ReactNode;
}) {
    const [view, setView] = useState<"list" | "details">("list");
    const [selectedPlugin, setSelectedPlugin] = useState<any>(null);
    const [allPlugins, setAllPlugins] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [installCounts, setInstallCounts] = useState<Map<string, number>>(new Map());
    const [searchQuery, setSearchQuery] = useState("");
    const [isSearching, setIsSearching] = useState(false);

    // @ts-ignore
    const { isFocused } = useTerminalFocus();

    // Filtered results
    const filteredPlugins = useMemo(() => {
        if (!searchQuery) return allPlugins;
        const q = searchQuery.toLowerCase();
        return allPlugins.filter(p =>
            p.entry.name.toLowerCase().includes(q) ||
            p.entry.description?.toLowerCase().includes(q) ||
            p.marketplaceName.toLowerCase().includes(q)
        );
    }, [allPlugins, searchQuery]);

    const [selectedIndex, setSelectedIndex] = useState(0);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [installingIds, setInstallingIds] = useState<Set<string>>(new Set());

    const pagination = useScroll({
        totalItems: filteredPlugins.length,
        selectedIndex,
        pageSize: 7
    });

    useEffect(() => {
        setSelectedIndex(0);
    }, [searchQuery]);

    // Initial Load
    useEffect(() => {
        async function load() {
            try {
                const marketplaceConfigs = await fetchMarketplaces();
                const { marketplaces } = await processMarketplaceData(marketplaceConfigs);

                const plugins: any[] = [];
                for (const m of marketplaces) {
                    if (m.data) {
                        for (const entry of m.data.plugins) {
                            plugins.push({
                                entry,
                                marketplaceName: m.name,
                                pluginId: `${entry.name}@${m.name}`,
                                isInstalled: false
                            });
                        }
                    }
                }

                // Sort by install counts if available
                const counts = await fetchPluginInstallCounts();
                setInstallCounts(counts);

                plugins.sort((a, b) => {
                    const countA = counts.get(a.pluginId) ?? 0;
                    const countB = counts.get(b.pluginId) ?? 0;
                    if (countA !== countB) return countB - countA;
                    return a.entry.name.localeCompare(b.entry.name);
                });

                setAllPlugins(plugins);

                if (targetPlugin) {
                    const match = plugins.find(p => p.entry.name === targetPlugin);
                    if (match) {
                        setSelectedPlugin(match);
                        setView("details");
                    }
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to load plugins");
            } finally {
                setIsLoading(false);
            }
        }
        load();
    }, [setError, targetPlugin]);

    const handleInstall = async () => {
        if (selectedIds.size === 0) return;
        const ids = Array.from(selectedIds);
        setInstallingIds(new Set(ids));

        let successCount = 0;
        for (const id of ids) {
            const plugin = allPlugins.find(p => p.pluginId === id);
            if (!plugin) continue;

            try {
                const res = await handlePluginAction({
                    pluginId: id,
                    entry: plugin.entry,
                    marketplaceName: plugin.marketplaceName,
                    scope: "user"
                });
                if (res.success) successCount++;
            } catch (err) {
                setError(`Failed to install ${plugin.entry.name}: ${err}`);
            }
        }

        if (successCount > 0) {
            logTelemetryEvent('tengu_plugins_installed_bulk', { count: successCount });
            if (onInstallComplete) await onInstallComplete();
            setViewState({ type: "menu", message: `Successfully installed ${successCount} plugin(s). Restart to apply.` });
        }
        setInstallingIds(new Set());
    };

    useInput((input, key) => {
        if (isSearching) {
            if (key.escape) {
                if (searchQuery) setSearchQuery("");
                else setIsSearching(false);
            } else if (key.return || key.upArrow || key.downArrow) {
                setIsSearching(false);
            } else if (key.backspace || key.delete) {
                setSearchQuery(prev => prev.slice(0, -1));
            } else if (input && !key.ctrl && !key.meta) {
                setSearchQuery(prev => prev + input);
            }
            return;
        }

        if (key.escape) {
            if (view === "details") {
                setView("list");
                setSelectedPlugin(null);
            } else {
                setViewState({ type: "menu" });
            }
            return;
        }

        if (view === "list") {
            if (input === "/") {
                setIsSearching(true);
                return;
            }

            if (key.upArrow || input === "k") {
                if (selectedIndex === 0) setIsSearching(true);
                else setSelectedIndex(prev => prev - 1);
            } else if (key.downArrow || input === "j") {
                if (selectedIndex < filteredPlugins.length - 1) setSelectedIndex(prev => prev + 1);
            } else if (input === " ") {
                const plugin = filteredPlugins[selectedIndex];
                if (plugin) {
                    const next = new Set(selectedIds);
                    if (next.has(plugin.pluginId)) next.delete(plugin.pluginId);
                    else next.add(plugin.pluginId);
                    setSelectedIds(next);
                }
            } else if (key.return) {
                const plugin = filteredPlugins[selectedIndex];
                if (plugin) {
                    setSelectedPlugin(plugin);
                    setView("details");
                }
            } else if (input === "i") {
                handleInstall();
            }
        }
    });

    if (isLoading) {
        return (
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                {children}
                <Text>Loading available plugins…</Text>
            </Box>
        );
    }

    if (view === "details" && selectedPlugin) {
        return (
            <PluginDetailsView
                plugin={selectedPlugin}
                onBack={() => setView("list")}
                onInstall={async (scope: string) => {
                    const res = await handlePluginAction({
                        pluginId: selectedPlugin.pluginId,
                        entry: selectedPlugin.entry,
                        marketplaceName: selectedPlugin.marketplaceName,
                        scope
                    });
                    if (res.success) {
                        if (onInstallComplete) await onInstallComplete();
                        setViewState({ type: "menu", message: res.message });
                    } else {
                        setError(res.message);
                    }
                }}
            >
                {children}
            </PluginDetailsView>
        );
    }

    const visibleItems = pagination.getVisibleItems(filteredPlugins);

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                {children}
                <Box marginBottom={1}>
                    <Text bold>Discover plugins </Text>
                    {pagination.needsPagination && (
                        <Text dimColor>({pagination.scrollPosition.current}/{pagination.scrollPosition.total})</Text>
                    )}
                </Box>

                <Box marginBottom={1}>
                    <Text color={isSearching ? "suggestion" : undefined}>
                        🔍 {searchQuery || (isSearching ? "" : "Search plugins...")}
                        {isSearching && "█"}
                    </Text>
                </Box>

                {filteredPlugins.length === 0 ? (
                    <MarketplaceStatusView reason={allPlugins.length === 0 ? "no-marketplaces-configured" : null} />
                ) : (
                    <>
                        {pagination.scrollPosition.canScrollUp && <Text dimColor>  {figures.arrowUp} more above</Text>}
                        {visibleItems.map((p: any, i: number) => {
                            const isFocusedItem = selectedIndex === pagination.toActualIndex(i);
                            const isSelected = selectedIds.has(p.pluginId);
                            const isInstalling = installingIds.has(p.pluginId);

                            return (
                                <Box key={p.pluginId} flexDirection="column" marginBottom={1}>
                                    <Box>
                                        <Text color={isFocusedItem && !isSearching ? "suggestion" : undefined}>
                                            {isFocusedItem && !isSearching ? figures.pointer : "  "}
                                        </Text>
                                        <Text>
                                            {isInstalling ? figures.ellipsis : (isSelected ? figures.radioOn : figures.radioOff)}
                                            {" "}{p.entry.name}
                                            <Text dimColor> · {p.marketplaceName}</Text>
                                            {installCounts.has(p.pluginId) && (
                                                <Text dimColor> · {installCounts.get(p.pluginId)} installs</Text>
                                            )}
                                        </Text>
                                    </Box>
                                    {p.entry.description && (
                                        <Box marginLeft={4}>
                                            <Text dimColor wrap="truncate-end">
                                                {p.entry.description}
                                            </Text>
                                        </Box>
                                    )}
                                </Box>
                            );
                        })}
                        {pagination.scrollPosition.canScrollDown && <Text dimColor>  {figures.arrowDown} more below</Text>}
                    </>
                )}
            </Box>
            <DiscoveryFooter hasSelection={selectedIds.size > 0} />
        </Box>
    );
}

function PluginDetailsView({ plugin, onBack, onInstall, children }: {
    plugin: {
        entry: any;
        marketplaceName: string;
    };
    onBack: () => void;
    onInstall: (scope: string) => void;
    children?: React.ReactNode;
}) {
    const actions = getHomepageActions(plugin.entry.homepage, getGithubUrl(plugin));
    const [focusedActionIndex, setFocusedActionIndex] = useState(0);

    useInput((input, key) => {
        if (key.upArrow || input === "k") {
            setFocusedActionIndex(prev => Math.max(0, prev - 1));
        } else if (key.downArrow || input === "j") {
            setFocusedActionIndex(prev => Math.min(actions.length - 1, prev + 1));
        } else if (key.return) {
            const action = actions[focusedActionIndex];
            if (action.action.startsWith("install-")) {
                const scope = action.action.split("-")[1];
                onInstall(scope);
            } else if (action.action === "homepage" && plugin.entry.homepage) {
                // In a real CLI would open browser
                console.log(`Opening ${plugin.entry.homepage}`);
            } else if (action.action === "back") {
                onBack();
            }
        }
    });

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                {children}
                <Box marginBottom={1}>
                    <Text bold>Plugin details</Text>
                </Box>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>{plugin.entry.name}</Text>
                    <Text dimColor>from {plugin.marketplaceName}</Text>
                    {plugin.entry.version && <Text dimColor>Version: {plugin.entry.version}</Text>}
                    {plugin.entry.description && (
                        <Box marginTop={1}>
                            <Text>{plugin.entry.description}</Text>
                        </Box>
                    )}
                </Box>
                <Box marginBottom={1}>
                    <Text color="warning">{figures.warning} </Text>
                    <Text dimColor italic>
                        Make sure you trust a plugin before installing. Claude Code does not verify third party MCP servers.
                    </Text>
                </Box>
                <Box flexDirection="column">
                    {actions.map((a: any, i: number) => (
                        <Text key={a.action} bold={i === focusedActionIndex}>
                            {i === focusedActionIndex ? "> " : "  "}
                            {a.label}
                        </Text>
                    ))}
                </Box>
            </Box>
            <Box marginTop={1} paddingLeft={1}>
                <Text dimColor><Text bold>Select:</Text> Enter · <Text bold>Back:</Text> Esc</Text>
            </Box>
        </Box>
    );
}

/**
 * Component for configuring plugin settings.
 * Derived from _X9 in chunk_570.ts.
 */
export function PluginConfigurationForm({
    pluginName,
    serverName,
    configSchema,
    onSave,
    onCancel
}: {
    pluginName: string;
    serverName: string;
    configSchema: Record<string, any>;
    onSave: (config: Record<string, any>) => void;
    onCancel: () => void;
}) {
    const fieldKeys = useMemo(() => Object.keys(configSchema), [configSchema]);
    const [currentFieldIndex, setCurrentFieldIndex] = useState(0);
    const [values, setValues] = useState<Record<string, string>>({});
    const [currentValue, setCurrentValue] = useState("");

    const currentKey = fieldKeys[currentFieldIndex];
    const currentField = currentKey ? configSchema[currentKey] : null;

    useInput((input, key) => {
        if (key.escape) {
            onCancel();
            return;
        }

        if (key.tab && currentFieldIndex < fieldKeys.length - 1) {
            if (currentKey) {
                setValues(prev => ({ ...prev, [currentKey]: currentValue }));
            }
            setCurrentFieldIndex(prev => prev + 1);
            setCurrentValue("");
            return;
        }

        if (key.return) {
            if (currentKey) {
                const updatedValues = { ...values, [currentKey]: currentValue };
                if (currentFieldIndex === fieldKeys.length - 1) {
                    const finalConfig: Record<string, any> = {};
                    for (const k of fieldKeys) {
                        const val = updatedValues[k] || "";
                        const schema = configSchema[k];
                        if (schema?.type === "number") {
                            const num = Number(val);
                            finalConfig[k] = isNaN(num) ? val : num;
                        } else if (schema?.type === "boolean") {
                            finalConfig[k] = val.toLowerCase() === "true" || val === "1";
                        } else {
                            finalConfig[k] = val;
                        }
                    }
                    onSave(finalConfig);
                } else {
                    setValues(updatedValues);
                    setCurrentFieldIndex(prev => prev + 1);
                    setCurrentValue("");
                }
            }
            return;
        }

        if (key.backspace || key.delete) {
            setCurrentValue(prev => prev.slice(0, -1));
            return;
        }

        if (input && !key.ctrl && !key.meta) {
            setCurrentValue(prev => prev + input);
        }
    });

    if (!currentField || !currentKey) return null;

    const isSensitive = currentField.sensitive === true;
    const isRequired = currentField.required === true;
    const displayText = isSensitive ? "*".repeat(currentValue.length) : currentValue;

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" gap={1} padding={1} borderStyle="round">
                <Text bold>Configure {serverName}</Text>
                <Box marginLeft={1}>
                    <Text dimColor>Plugin: {pluginName}</Text>
                </Box>
                <Box marginTop={1} flexDirection="column">
                    <Text bold>
                        {currentField.title || currentKey}
                        {isRequired && <Text color="error"> *</Text>}
                    </Text>
                    {currentField.description && (
                        <Text dimColor>{currentField.description}</Text>
                    )}
                    <Box marginTop={1}>
                        <Text>{figures.pointer} </Text>
                        <Text>{displayText}</Text>
                        <Text>█</Text>
                    </Box>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>Field {currentFieldIndex + 1} of {fieldKeys.length}</Text>
                </Box>
                {currentFieldIndex < fieldKeys.length - 1 ? (
                    <Text dimColor>Tab: Next field · Enter: Save and continue</Text>
                ) : (
                    <Text dimColor>Enter: Save configuration</Text>
                )}
            </Box>
            <Box marginLeft={3}>
                <Text dimColor>Esc to cancel</Text>
            </Box>
        </Box>
    );
}

/**
 * Lists agent descriptions from a plugin's directory.
 * Derived from TX9 in chunk_570.ts.
 */
export async function listPluginAgents(dirPath: string): Promise<string[]> {
    try {
        const entries = await fs.readdir(dirPath, { withFileTypes: true });
        return entries
            .filter(e => e.isFile() && e.name.endsWith(".md"))
            .map(e => path.basename(e.name, ".md"));
    } catch (err) {
        return [];
    }
}
