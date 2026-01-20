
// Logic from chunk_569.ts (Plugin Marketplace & Details)

import React, { useState, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { usePagination } from "./MarketplaceManager.js";
import { formatInstallCount } from "../../services/plugin/PluginStatsService.js";
import { logTelemetryEvent as logEvent } from "../../services/telemetry/telemetryInit.js";
import { useTheme } from "../../services/terminal/themeManager.js"; // Stub
import { openUrl } from "../../utils/shared/XDGUtils.js"; // F5
import { getThemeStyle } from "../../services/terminal/themeManager.js"; // yX

// --- Stubs/Placeholders for missing logic ---
import {
    validatePluginId, // Vz
    parsePluginId,    // ha
    getPluginSource,  // fTA
    getMarketplaceIssues,  // OWA
    getPluginFromMarketplace as fetchPluginFromMarketplace, // W$
    fetchMarketplaces, // t3
    processMarketplaceData, // oBA
    fetchPluginInstallCounts as fetchInstallCounts, // aH1
    installPlugin, // Dz
    getConfig, // gB
    updateConfig, // nB
    clearPluginCache, // sZ
    formatFailedInstalls, // LWA
    handlePluginAction, // h51
    getHomepageActions, // CFA
    getGithubUrl, // zFA
    loadMarketplace,
} from "../../services/mcp/PluginManager.js";

// -- Footer Component (wX9)
function PluginListFooter({ hasSelection }: { hasSelection: boolean }) {
    if (hasSelection) {
        return (
            <Box marginTop={1} paddingLeft={1}>
                <Text dimColor>Select: <Text bold>Space</Text> · Install: <Text bold>i</Text> · Cancel: <Text bold>Esc</Text></Text>
            </Box>
        );
    }
    return (
        <Box marginTop={1} paddingLeft={1}>
            <Text dimColor>Select: <Text bold>Enter</Text> · Back: <Text bold>Esc</Text></Text>
        </Box>
    );
}

export function PluginMarketplaceView({
    error: initialError,
    setError,
    result,
    setResult,
    setViewState,
    onInstallComplete,
    targetMarketplace,
    targetPlugin,
    children
}: any) {
    const [view, setView] = useState("marketplace-list");
    const [selectedMarketplace, setSelectedMarketplace] = useState<string | null>(null);
    const [selectedPlugin, setSelectedPlugin] = useState<any | null>(null);
    const [marketplaces, setMarketplaces] = useState<any[]>([]);
    const [plugins, setPlugins] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [installCounts, setInstallCounts] = useState<Map<string, number> | null>(null);
    const [marketplaceListIndex, setMarketplaceListIndex] = useState(0);
    const [selectedPluginIds, setSelectedPluginIds] = useState<Set<string>>(new Set());
    const [installedPluginIds, setInstalledPluginIds] = useState<Set<string>>(new Set());

    // Pagination
    const pagination = usePagination({
        totalItems: plugins.length,
        selectedIndex: marketplaceListIndex // Re-using index state for plugins due to logic shared
    });

    const [detailsIndex, setDetailsIndex] = useState(0);
    const [isInstalling, setIsInstalling] = useState(false);
    const [actionError, setActionError] = useState<string | null>(null);
    const [warningMessage, setWarningMessage] = useState<string | null>(null);

    // Initial Load
    useEffect(() => {
        async function load() {
            setIsLoading(true);
            try {
                const configs = await fetchMarketplaces();
                const { marketplaces: loaded, failures } = await processMarketplaceData(configs);

                const mapped = [];
                for (const { name, config, data } of loaded) {
                    if (data) {
                        // Stub installed count logic
                        let installedCount = 0; // data.plugins.filter...
                        mapped.push({
                            name,
                            totalPlugins: data.plugins.length,
                            installedCount,
                            source: getPluginSource((config as any).source)
                        });
                    }
                }

                mapped.sort((a, b) => {
                    if (a.name === "claude-plugin-directory") return -1;
                    if (b.name === "claude-plugin-directory") return 1;
                    return 0;
                });
                setMarketplaces(mapped);

                const successCount = loaded.filter(m => m.data !== null).length;
                const warning = getMarketplaceIssues(failures, successCount);
                if (warning) {
                    if (warning.type === "warning") setWarningMessage(warning.message + ". Showing available marketplaces.");
                    else throw new Error(warning.message);
                }

                // Auto-selection logic
                if (mapped.length === 1 && !targetMarketplace && !targetPlugin) {
                    const m = mapped[0];
                    if (m) {
                        setSelectedMarketplace(m.name);
                        setView("plugin-list");
                    }
                }

                if (targetPlugin) {
                    // Logic to find specific plugin across marketplaces
                    // Stubbed for now
                    setError(`Plugin "${targetPlugin}" not found logic stubbed`);
                } else if (targetMarketplace) {
                    if (mapped.some(m => m.name === targetMarketplace)) {
                        setSelectedMarketplace(targetMarketplace);
                        setView("plugin-list");
                    } else {
                        setError(`Marketplace "${targetMarketplace}" not found`);
                    }
                }

            } catch (err: any) {
                setError(err instanceof Error ? err.message : "Failed to load marketplaces");
            } finally {
                setIsLoading(false);
            }
        }
        load();
    }, [setError, targetMarketplace, targetPlugin]);

    // Plugin List Load
    useEffect(() => {
        if (!selectedMarketplace) return;
        async function loadPlugins(mpName: string) {
            setIsLoading(true);
            try {
                const mpData = await loadMarketplace(mpName);
                if (!mpData) throw new Error(`Failed to load marketplace: ${mpName}`);

                const mappedPlugins = [];
                for (const p of mpData.plugins) {
                    const pid = `${p.name}@${mpName}`;
                    // isInstalled logic stub
                    mappedPlugins.push({
                        entry: p,
                        marketplaceName: mpName,
                        pluginId: pid,
                        isInstalled: false
                    });
                }

                try {
                    const counts = await fetchInstallCounts();
                    setInstallCounts(counts);
                    if (counts) {
                        mappedPlugins.sort((a, b) => {
                            const ca = counts.get(a.pluginId) ?? 0;
                            const cb = counts.get(b.pluginId) ?? 0;
                            if (ca !== cb) return cb - ca;
                            return a.entry.name.localeCompare(b.entry.name);
                        });
                    } else {
                        mappedPlugins.sort((a, b) => a.entry.name.localeCompare(b.entry.name));
                    }
                } catch (e: any) {
                    // fall back
                    mappedPlugins.sort((a, b) => a.entry.name.localeCompare(b.entry.name));
                }

                setPlugins(mappedPlugins);
                setMarketplaceListIndex(0); // Reset index
                setSelectedPluginIds(new Set());
            } catch (e: any) {
                setError(e instanceof Error ? e.message : "Failed to load plugins");
            } finally {
                setIsLoading(false);
            }
        }
        loadPlugins(selectedMarketplace);
    }, [selectedMarketplace, setError]);


    // Install Action
    const handleInstall = async () => {
        if (selectedPluginIds.size === 0) return;
        const targets = plugins.filter(p => selectedPluginIds.has(p.pluginId));
        setInstalledPluginIds(new Set(targets.map(p => p.pluginId)));

        let success = 0;
        let failures = 0;
        const failedList = [];


        for (const p of targets) {
            try {
                if (typeof p.entry.source !== "string") await installPlugin(p.pluginId, "user");

                // Update config stub
                const config = getConfig("userSettings") as any;
                // ... update logic
                success++;
                logEvent("tengu_plugin_installed", {
                    plugin_id: p.pluginId,
                    marketplace_name: p.marketplaceName
                });
            } catch (e: any) {
                failures++;
                const msg = e instanceof Error ? e.message : String(e);
                failedList.push({ name: p.entry.name, reason: msg });
            }
        }

        setInstalledPluginIds(new Set());
        setSelectedPluginIds(new Set());
        // clearPluginCache();

        if (failures === 0) {
            setResult(`✓ Installed ${success} plugin${success !== 1 ? "s" : ""}. Restart Claude Code to load new plugins.`);
        } else if (success === 0) {
            setError(`Failed to install: ${formatFailedInstalls(failedList, true)}`);
        } else {
            setResult(`✓ Installed ${success} of ${success + failures} plugins. Failed: ${formatFailedInstalls(failedList, false)}. Restart...`);
        }

        if (success > 0 && onInstallComplete) await onInstallComplete();
        setViewState({ type: "menu" });
    };

    // Single Details Action (Install/etc)
    const handleDetailsAction = async (plugin: any, scope = "user") => {
        setIsInstalling(true);
        setActionError(null);

        const res = await handlePluginAction({
            pluginId: plugin.pluginId,
            entry: plugin.entry,
            marketplaceName: plugin.marketplaceName,
            scope
        });

        if (res.success) {
            setResult(res.message);
            if (onInstallComplete) await onInstallComplete();
            setViewState({ type: "menu" });
        } else {
            setIsInstalling(false);
            setActionError(res.message);
        }
    };

    // Input Handling
    useInput((input, key) => {
        if (input === 'm' && (view === "marketplace-list" || view === "plugin-list")) {
            setViewState({ type: "manage-marketplaces" });
            return;
        }
        if (key.escape) {
            if (view === "plugin-list") {
                if (targetMarketplace) {
                    setViewState({ type: "manage-marketplaces", targetMarketplace });
                } else {
                    setView("marketplace-list");
                    setSelectedMarketplace(null);
                    setSelectedPluginIds(new Set());
                }
            } else if (view === "plugin-details") {
                setView("plugin-list");
                setSelectedPlugin(null);
            } else {
                setViewState({ type: "menu" });
            }
            return;
        }

        if (view === "marketplace-list") {
            if ((key.upArrow || input === 'k') && marketplaceListIndex > 0) setMarketplaceListIndex(marketplaceListIndex - 1);
            else if ((key.downArrow || input === 'j') && marketplaceListIndex < marketplaces.length - 1) setMarketplaceListIndex(marketplaceListIndex + 1);
            else if (key.return) {
                const m = marketplaces[marketplaceListIndex];
                if (m) {
                    setSelectedMarketplace(m.name);
                    setView("plugin-list");
                }
            }
        } else if (view === "plugin-list") {
            // Pagination logic from c (usePagination return)
            // Simplified input handling here mapping to usePagination logic
            const total = plugins.length;
            // Tab cycling logic ommitted for brevity/complexity match

            if ((key.upArrow || input === 'k')) {
                if (marketplaceListIndex > 0) pagination.handleSelectionChange(marketplaceListIndex - 1, setMarketplaceListIndex);
            } else if ((key.downArrow || input === 'j')) {
                if (marketplaceListIndex < total - 1) pagination.handleSelectionChange(marketplaceListIndex + 1, setMarketplaceListIndex);
            } else if (input === ' ') {
                if (marketplaceListIndex < plugins.length) {
                    const p = plugins[marketplaceListIndex];
                    if (p && !p.isInstalled) {
                        const next = new Set(selectedPluginIds);
                        if (next.has(p.pluginId)) next.delete(p.pluginId);
                        else next.add(p.pluginId);
                        setSelectedPluginIds(next);
                    }
                }
            } else if (key.return) {
                if (marketplaceListIndex === plugins.length && selectedPluginIds.size > 0) {
                    handleInstall();
                } else if (marketplaceListIndex < plugins.length) {
                    const p = plugins[marketplaceListIndex];
                    if (p) {
                        if (p.isInstalled) {
                            setViewState({
                                type: "manage-plugins",
                                targetPlugin: p.entry.name,
                                targetMarketplace: p.marketplaceName
                            });
                        } else {
                            setSelectedPlugin(p);
                            setView("plugin-details");
                            setDetailsIndex(0);
                            setActionError(null);
                        }
                    }
                }
            } else if (input === 'i' && selectedPluginIds.size > 0) {
                handleInstall();
            }
        } else if (view === "plugin-details" && selectedPlugin) {
            const homepage = selectedPlugin.entry.homepage;
            const githubUrl = getGithubUrl(selectedPlugin);
            const actions = getHomepageActions(homepage, githubUrl);

            if ((key.upArrow || input === 'k') && detailsIndex > 0) setDetailsIndex(detailsIndex - 1);
            else if ((key.downArrow || input === 'j') && detailsIndex < actions.length - 1) setDetailsIndex(detailsIndex + 1);
            else if (key.return) {
                const action = actions[detailsIndex]?.action;
                if (action === "install-user") handleDetailsAction(selectedPlugin, "user");
                else if (action === "install-project") handleDetailsAction(selectedPlugin, "project");
                else if (action === "install-local") handleDetailsAction(selectedPlugin, "local");
                else if (action === "homepage" && homepage) openUrl(homepage);
                else if (action === "github" && githubUrl) openUrl(`https://github.com/${githubUrl}`);
                else if (action === "back") {
                    setView("plugin-list");
                    setSelectedPlugin(null);
                }
            }
        }
    });

    if (isLoading) {
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Text>Loading…</Text>
                </Box>
            </Box>
        );
    }

    if (initialError) {
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Text color="red">{initialError}</Text>
                </Box>
            </Box>
        );
    }

    // START RENDER LOGIC

    // Fix error references
    const error = initialError || actionError;

    if (view === "marketplace-list") {
        if (marketplaces.length === 0) {
            return (
                <Box flexDirection="column">
                    <Box flexDirection="column" paddingX={1} borderStyle="round">
                        {children}
                        <Box marginBottom={1}><Text bold>Select marketplace</Text></Box>
                        <Text>No marketplaces configured.</Text>
                        <Text dimColor>Add a marketplace first using 'Add marketplace'.</Text>
                    </Box>
                    <Box marginTop={1} paddingLeft={1}><Text dimColor>Esc to go back</Text></Box>
                </Box>
            );
        }

        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Box marginBottom={1}><Text bold>Select marketplace</Text></Box>
                    {warningMessage && (
                        <Box marginBottom={1} flexDirection="column">
                            <Text color="yellow">{figures.warning} {warningMessage}</Text>
                        </Box>
                    )}
                    {marketplaces.map((m, i) => (
                        <Box key={m.name} flexDirection="column" marginBottom={i < marketplaces.length - 1 ? 1 : 0}>
                            <Box>
                                <Text color={marketplaceListIndex === i ? "suggestion" : undefined}>
                                    {marketplaceListIndex === i ? figures.pointer : " "} {m.name}
                                </Text>
                            </Box>
                            <Box marginLeft={2}>
                                <Text dimColor>
                                    {m.totalPlugins} plugin{m.totalPlugins !== 1 ? "s" : ""} available
                                    {m.installedCount > 0 && ` · ${m.installedCount} already installed`}
                                    {m.source && ` · ${m.source}`}
                                </Text>
                            </Box>
                        </Box>
                    ))}
                </Box>
                <Box paddingLeft={1}>
                    <Text dimColor italic>Enter to select · m: manage marketplaces · Esc to go back</Text>
                </Box>
            </Box>
        );
    }

    if (view === "plugin-details" && selectedPlugin) {
        const homepage = selectedPlugin.entry.homepage;
        const githubUrl = getGithubUrl(selectedPlugin);
        const actions = getHomepageActions(homepage, githubUrl);

        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Box marginBottom={1}><Text bold>Plugin Details</Text></Box>

                    <Box flexDirection="column" marginBottom={1}>
                        <Text bold>{selectedPlugin.entry.name}</Text>
                        {selectedPlugin.entry.version && <Text dimColor>Version: {selectedPlugin.entry.version}</Text>}
                        {selectedPlugin.entry.description && (
                            <Box marginTop={1}><Text>{selectedPlugin.entry.description}</Text></Box>
                        )}
                        {selectedPlugin.entry.author && (
                            <Box marginTop={1}>
                                <Text dimColor>By: {typeof selectedPlugin.entry.author === "string" ? selectedPlugin.entry.author : selectedPlugin.entry.author?.name}</Text>
                            </Box>
                        )}
                    </Box>

                    <Box flexDirection="column" marginBottom={1}>
                        <Text bold>Will install:</Text>
                        {selectedPlugin.entry.commands && (
                            <Text dimColor>• Commands: {Array.isArray(selectedPlugin.entry.commands) ? selectedPlugin.entry.commands.join(", ") : Object.keys(selectedPlugin.entry.commands).join(", ")}</Text>
                        )}
                        {selectedPlugin.entry.agents && (
                            <Text dimColor>• Agents: {Array.isArray(selectedPlugin.entry.agents) ? selectedPlugin.entry.agents.join(", ") : Object.keys(selectedPlugin.entry.agents).join(", ")}</Text>
                        )}
                        {selectedPlugin.entry.hooks && (
                            <Text dimColor>• Hooks: {Object.keys(selectedPlugin.entry.hooks).join(", ")}</Text>
                        )}
                        {/* Fallback for components not explicit */}
                        {!selectedPlugin.entry.commands && !selectedPlugin.entry.agents && (
                            <Text dimColor>• Components will be discovered at installation</Text>
                        )}
                    </Box>

                    <Box marginBottom={1}>
                        <Text color="claude">{figures.warning} </Text>
                        <Text dimColor italic>
                            Make sure you trust a plugin before installing, updating, or using it. Anthropic does not control what MCP servers, files, or other software are included in plugins and cannot verify that they will work as intended or that they won't change. See each plugin's homepage for more information.
                        </Text>
                    </Box>

                    {actionError && (
                        <Box marginBottom={1}>
                            <Text color="red">Error: {actionError}</Text>
                        </Box>
                    )}

                    <Box flexDirection="column">
                        {actions.map((act: any, i: number) => (
                            <Box key={act.action}>
                                <Text>{detailsIndex === i ? "> " : "  "}</Text>
                                <Text bold={detailsIndex === i}>
                                    {isInstalling && act.action === "install" ? "Installing…" : act.label}
                                </Text>
                            </Box>
                        ))}
                    </Box>
                </Box>

                <Box marginTop={1} paddingLeft={1}>
                    <Text dimColor>
                        <Text bold>Select:</Text> Enter · <Text bold>Back:</Text> Esc
                    </Text>
                </Box>
            </Box>
        );
    }

    if (plugins.length === 0) {
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Box marginBottom={1}><Text bold>Install plugins</Text></Box>
                    <Text dimColor>No new plugins available to install.</Text>
                </Box>
                <Box marginLeft={3}><Text dimColor italic>Esc to go back</Text></Box>
            </Box>
        );
    }

    const visiblePlugins = pagination.getVisibleItems(plugins);

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                {children}
                <Box marginBottom={1}>
                    <Text bold>Install Plugins</Text>
                    {pagination.needsPagination && (
                        <Text dimColor> ({pagination.scrollPosition.current}/{pagination.scrollPosition.total})</Text>
                    )}
                </Box>
                {pagination.scrollPosition.canScrollUp && (
                    <Box><Text dimColor> {figures.arrowUp} more above</Text></Box>
                )}
                {visiblePlugins.map((p, i) => {
                    const actualIndex = pagination.toActualIndex(i);
                    const isSelected = marketplaceListIndex === actualIndex;
                    const isMarked = selectedPluginIds.has(p.pluginId);
                    const isInstallingItem = installedPluginIds.has(p.pluginId);
                    const isLast = i === visiblePlugins.length - 1;

                    return (
                        <Box key={p.pluginId} flexDirection="column" marginBottom={isLast && !error ? 0 : 1}>
                            <Box>
                                <Text color={isSelected ? "suggestion" : undefined}>
                                    {isSelected ? figures.pointer : " "}
                                </Text>
                                <Text color={p.isInstalled ? "success" : undefined}>
                                    {p.isInstalled ? figures.tick : isInstallingItem ? figures.ellipsis : isMarked ? figures.radioOn : figures.radioOff} {p.entry.name}
                                    {p.entry.category && <Text dimColor> [{p.entry.category}]</Text>}
                                    {p.isInstalled && <Text dimColor> (installed)</Text>}
                                    {installCounts && <Text dimColor> · {formatInstallCount(installCounts.get(p.pluginId) ?? 0)} installs</Text>}
                                </Text>
                            </Box>
                            {p.entry.description && (
                                <Box marginLeft={4}>
                                    <Text dimColor>
                                        {p.entry.description.length > 60 ? p.entry.description.substring(0, 57) + "..." : p.entry.description}
                                    </Text>
                                    {p.entry.version && <Text dimColor> · v{p.entry.version}</Text>}
                                </Box>
                            )}
                        </Box>
                    );
                })}
                {pagination.scrollPosition.canScrollDown && (
                    <Box><Text dimColor> {figures.arrowDown} more below</Text></Box>
                )}
                {error && (
                    <Box marginTop={1}><Text color="error">{figures.cross} {error}</Text></Box>
                )}
            </Box>
            <PluginListFooter hasSelection={selectedPluginIds.size > 0} />
        </Box>
    );
}
