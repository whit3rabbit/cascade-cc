// Logic from chunk_567.ts (Marketplace Manager & Pagination)

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { readMarketplaceConfigFile, writeMarketplaceConfigFile } from "../../services/marketplace/MarketplaceConfig.js";
import { loadConfiguredMarketplaces } from "../../services/marketplace/MarketplaceLoader.js";
import { MarketplaceService } from "../../services/marketplace/MarketplaceService.js";
import { getSettings, updateSettings } from "../../services/terminal/settings.js";
import { InstalledPluginStore } from "../../services/plugin/InstalledPluginStore.js";
import { Select } from "../shared/Select.js";

export { Select };

type ExitState = { pending: boolean; keyName: string };
type ViewState =
    | { type: "menu" }
    | { type: "add-marketplace" }
    | { type: "browse-marketplace"; targetMarketplace: string };

type InstalledPlugin = {
    id?: string;
    name: string;
    source: string;
    manifest: { description?: string };
};

type MarketplaceInfo = {
    name: string;
    source: string;
    lastUpdated?: string;
    pluginCount?: number;
    installedPlugins?: InstalledPlugin[];
    pendingUpdate: boolean;
    pendingRemove: boolean;
    autoUpdate?: boolean;
};

type MarketplaceMenuOption = {
    label: string;
    value: "browse" | "update" | "toggle-auto-update" | "remove";
    secondaryLabel?: string;
};

type ManageMarketplacesProps = {
    setViewState: (state: ViewState) => void;
    error?: string | null;
    setError?: (message: string | null) => void;
    setResult: (message: string) => void;
    exitState: ExitState;
    onManageComplete?: () => Promise<void> | void;
    targetMarketplace?: string;
    action?: "update" | "remove";
    children?: React.ReactNode;
};

type MarketplaceFailureSummary = { type: "warning" | "error"; message: string } | null;

const trackEvent = (name: string, payload?: Record<string, any>) => {
    // Telemetry logic
};


function formatMarketplaceSource(source: any): string {
    if (!source) return "Unknown source";
    if (typeof source === "string") return source;
    if (source.source === "github" && source.repo) {
        return source.ref ? `github.com/${source.repo}#${source.ref}` : `github.com/${source.repo}`;
    }
    if (source.source === "git" && source.url) {
        return source.ref ? `${source.url}#${source.ref}` : source.url;
    }
    if (source.source === "url" && source.url) return source.url;
    if ((source.source === "file" || source.source === "directory") && source.path) return source.path;
    return JSON.stringify(source);
}

function getMarketplaceConfig(): Record<string, any> {
    try {
        return readMarketplaceConfigFile();
    } catch {
        return {};
    }
}

async function getInstalledPlugins(): Promise<{ enabled: InstalledPlugin[]; disabled: InstalledPlugin[] }> {
    const settings = getSettings("userSettings") as { enabledPlugins?: Record<string, boolean | string[]> };
    const enabledMap = settings?.enabledPlugins ?? {};
    const store = InstalledPluginStore.getAllInstalledPlugins();
    const entries = Object.entries(store.plugins ?? {});

    const all = entries.map(([pluginId, installs]) => {
        const name = pluginId.split("@")[0];
        const description = Array.isArray(installs) && installs.length > 0 ? installs[0].description : "";
        return {
            id: pluginId,
            name,
            source: pluginId,
            manifest: { description }
        } as InstalledPlugin;
    });

    const enabled: InstalledPlugin[] = [];
    const disabled: InstalledPlugin[] = [];

    for (const plugin of all) {
        const flag = enabledMap[plugin.id ?? plugin.name];
        const isEnabled = Array.isArray(flag) ? flag.length > 0 : Boolean(flag);
        if (isEnabled) enabled.push(plugin);
        else disabled.push(plugin);
    }

    return { enabled, disabled };
}

function summarizeMarketplaceFailures(failures: any[], loadedCount: number): MarketplaceFailureSummary {
    if (!failures || failures.length === 0) return null;
    const total = failures.length + loadedCount;
    const suffix = failures.length === 1 ? "marketplace" : "marketplaces";
    return {
        type: "warning",
        message: `Failed to load ${failures.length} of ${total} ${suffix}`
    };
}

function isAutoUpdateLocked(): boolean {
    return false;
}

async function setMarketplaceAutoUpdate(name: string, enabled: boolean): Promise<void> {
    const config = getMarketplaceConfig();
    const entry = config[name];
    if (!entry) throw new Error(`Marketplace '${name}' not found`);
    entry.autoUpdate = enabled;
    writeMarketplaceConfigFile(config);
}

const resetMarketplaceCache = () => {
    // Cache reset logic
};


export function MarketplaceFooter({
    exitState,
    hasPendingActions
}: {
    exitState: ExitState;
    hasPendingActions: boolean;
}) {
    const items: string[] = [];
    if (exitState?.pending) {
        items.push(`Press ${exitState.keyName} again to go back`);
    } else {
        items.push(`${figures.arrowUp}${figures.arrowDown}`);
        if (hasPendingActions) {
            items.push("Enter to apply changes");
        } else {
            items.push("Enter to select");
            items.push("u update");
            items.push("r remove");
        }
        items.push(hasPendingActions ? "Esc to cancel" : "Esc to go back");
    }

    return (
        <Box marginLeft={3}>
            <Text dimColor italic>{items.join(" · ")}</Text>
        </Box>
    );
}

export function ManageMarketplaces({
    setViewState,
    error,
    setError,
    setResult,
    exitState,
    onManageComplete,
    targetMarketplace,
    action,
    children
}: ManageMarketplacesProps) {
    const [marketplaces, setMarketplaces] = useState<MarketplaceInfo[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [isProcessing, setIsProcessing] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    const [progressMessage, setProgressMessage] = useState<string | null>(null);
    const [mode, setMode] = useState<"list" | "details" | "confirm-remove">("list");
    const [selectedMarketplace, setSelectedMarketplace] = useState<MarketplaceInfo | null>(null);
    const [detailsIndex, setDetailsIndex] = useState(0);
    const didAutoSelect = useRef(false);

    useEffect(() => {
        async function loadMarketplaces() {
            try {
                const config = getMarketplaceConfig();
                const { enabled, disabled } = await getInstalledPlugins();
                const installed = [...enabled, ...disabled];
                const { marketplaces: loaded, failures } = await loadConfiguredMarketplaces(config);
                const next: MarketplaceInfo[] = [];

                for (const { name, config: configEntry, data } of loaded) {
                    const installedPlugins = installed.filter((plugin) => plugin.source.endsWith(`@${name}`));
                    next.push({
                        name,
                        source: formatMarketplaceSource(configEntry.source),
                        lastUpdated: configEntry.lastUpdated,
                        pluginCount: data?.plugins?.length,
                        installedPlugins,
                        pendingUpdate: false,
                        pendingRemove: false,
                        autoUpdate: Boolean(configEntry.autoUpdate)
                    });
                }

                next.sort((a, b) => {
                    if (a.name === "claude-plugin-directory") return -1;
                    if (b.name === "claude-plugin-directory") return 1;
                    return a.name.localeCompare(b.name);
                });

                setMarketplaces(next);
                const loadedCount = loaded.filter((entry) => entry.data !== null).length;
                const summary = summarizeMarketplaceFailures(failures, loadedCount);
                if (summary) {
                    if (summary.type === "warning") setErrorMessage(summary.message);
                    else throw new Error(summary.message);
                }

                if (targetMarketplace && !didAutoSelect.current && !error) {
                    didAutoSelect.current = true;
                    const index = next.findIndex((entry) => entry.name === targetMarketplace);
                    if (index >= 0) {
                        const match = next[index];
                        if (action) {
                            setSelectedIndex(index + 1);
                            const updated = [...next];
                            if (action === "update") updated[index].pendingUpdate = true;
                            if (action === "remove") updated[index].pendingRemove = true;
                            setMarketplaces(updated);
                            setTimeout(() => {
                                void applyChanges(updated);
                            }, 100);
                        } else if (match) {
                            setSelectedIndex(index + 1);
                            setSelectedMarketplace(match);
                            setMode("details");
                        }
                    } else if (setError) {
                        setError(`Marketplace not found: ${targetMarketplace}`);
                    }
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : "Failed to load marketplaces";
                if (setError) setError(message);
                setErrorMessage(message);
            } finally {
                setIsLoading(false);
            }
        }

        loadMarketplaces();
    }, [targetMarketplace, action, error, setError]);

    const hasPendingChanges = useCallback(() => {
        return marketplaces.some((entry) => entry.pendingUpdate || entry.pendingRemove);
    }, [marketplaces]);

    const pendingCounts = useCallback(() => {
        const updateCount = marketplaces.filter((entry) => entry.pendingUpdate).length;
        const removeCount = marketplaces.filter((entry) => entry.pendingRemove).length;
        return { updateCount, removeCount };
    }, [marketplaces]);

    const applyChanges = useCallback(
        async (overrideList?: MarketplaceInfo[]) => {
            const working = overrideList ?? marketplaces;
            const wasDetailsView = mode === "details";

            setIsProcessing(true);
            setErrorMessage(null);
            setSuccessMessage(null);
            setProgressMessage(null);

            try {
                const settings = getSettings("userSettings") as { enabledPlugins?: Record<string, boolean> };
                let updatedCount = 0;
                let removedCount = 0;

                for (const entry of working) {
                    if (entry.pendingRemove) {
                        if (entry.installedPlugins && entry.installedPlugins.length > 0) {
                            const enabledPlugins = { ...(settings?.enabledPlugins ?? {}) };
                            for (const plugin of entry.installedPlugins) {
                                enabledPlugins[plugin.source] = false;
                            }
                            updateSettings("userSettings", { enabledPlugins });
                        }
                        await MarketplaceService.removeMarketplace(entry.name);
                        removedCount += 1;
                        trackEvent("tengu_marketplace_removed", {
                            marketplace_name: entry.name,
                            plugins_uninstalled: entry.installedPlugins?.length || 0
                        });
                        continue;
                    }

                    if (entry.pendingUpdate) {
                        await MarketplaceService.refreshMarketplace(entry.name, (message) => {
                            setProgressMessage(message);
                        });
                        updatedCount += 1;
                        trackEvent("tengu_marketplace_updated", {
                            marketplace_name: entry.name
                        });
                    }
                }

                resetMarketplaceCache();
                if (onManageComplete) await onManageComplete();

                const config = getMarketplaceConfig();
                const { enabled, disabled } = await getInstalledPlugins();
                const installed = [...enabled, ...disabled];
                const { marketplaces: loaded } = await loadConfiguredMarketplaces(config);
                const next: MarketplaceInfo[] = [];

                for (const { name, config: configEntry, data } of loaded) {
                    const installedPlugins = installed.filter((plugin) => plugin.source.endsWith(`@${name}`));
                    next.push({
                        name,
                        source: formatMarketplaceSource(configEntry.source),
                        lastUpdated: configEntry.lastUpdated,
                        pluginCount: data?.plugins?.length,
                        installedPlugins,
                        pendingUpdate: false,
                        pendingRemove: false,
                        autoUpdate: Boolean(configEntry.autoUpdate)
                    });
                }

                next.sort((a, b) => {
                    if (a.name === "claude-plugin-directory") return -1;
                    if (b.name === "claude-plugin-directory") return 1;
                    return a.name.localeCompare(b.name);
                });

                setMarketplaces(next);

                if (wasDetailsView && selectedMarketplace) {
                    const refreshed = next.find((entry) => entry.name === selectedMarketplace.name);
                    if (refreshed) setSelectedMarketplace(refreshed);
                }

                const summaryParts: string[] = [];
                if (updatedCount > 0) summaryParts.push(`Updated ${updatedCount} marketplace${updatedCount > 1 ? "s" : ""}`);
                if (removedCount > 0) summaryParts.push(`Removed ${removedCount} marketplace${removedCount > 1 ? "s" : ""}`);

                if (summaryParts.length > 0) {
                    const summary = `${figures.tick} ${summaryParts.join(", ")}`;
                    if (wasDetailsView) {
                        setSuccessMessage(summary);
                    } else {
                        setResult(summary);
                        setTimeout(() => {
                            setViewState({ type: "menu" });
                        }, 2000);
                    }
                } else if (!wasDetailsView) {
                    setViewState({ type: "menu" });
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                setErrorMessage(message);
                if (setError) setError(message);
            } finally {
                setIsProcessing(false);
                setProgressMessage(null);
            }
        },
        [marketplaces, mode, onManageComplete, selectedMarketplace, setError, setResult, setViewState]
    );

    const confirmRemove = useCallback(async () => {
        if (!selectedMarketplace) return;
        const next = marketplaces.map((entry) =>
            entry.name === selectedMarketplace.name ? { ...entry, pendingRemove: true } : entry
        );
        setMarketplaces(next);
        await applyChanges(next);
    }, [marketplaces, selectedMarketplace, applyChanges]);

    const buildOptions = useCallback((entry?: MarketplaceInfo | null): MarketplaceMenuOption[] => {
        if (!entry) return [];
        const options: MarketplaceMenuOption[] = [
            {
                label: `Browse plugins (${entry.pluginCount ?? 0})`,
                value: "browse"
            },
            {
                label: "Update marketplace",
                secondaryLabel: entry.lastUpdated
                    ? `(last updated ${new Date(entry.lastUpdated).toLocaleDateString()})`
                    : undefined,
                value: "update"
            }
        ];
        if (!isAutoUpdateLocked()) {
            options.push({
                label: entry.autoUpdate ? "Disable auto-update" : "Enable auto-update",
                value: "toggle-auto-update"
            });
        }
        options.push({ label: "Remove marketplace", value: "remove" });
        return options;
    }, []);

    const toggleAutoUpdate = useCallback(async (entry: MarketplaceInfo) => {
        const nextValue = !entry.autoUpdate;
        try {
            await setMarketplaceAutoUpdate(entry.name, nextValue);
            setMarketplaces((prev) =>
                prev.map((item) => (item.name === entry.name ? { ...item, autoUpdate: nextValue } : item))
            );
            setSelectedMarketplace((prev) => (prev ? { ...prev, autoUpdate: nextValue } : prev));
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Failed to update setting");
        }
    }, []);

    useInput((input, key) => {
        if (isProcessing) return;

        if (key.escape) {
            if (mode === "details" || mode === "confirm-remove") {
                setMode("list");
                setDetailsIndex(0);
                return;
            }
            if (hasPendingChanges()) {
                setMarketplaces((prev) =>
                    prev.map((entry) => ({
                        ...entry,
                        pendingUpdate: false,
                        pendingRemove: false
                    }))
                );
                setSelectedIndex(0);
            } else {
                setViewState({ type: "menu" });
            }
            return;
        }

        if (mode === "list") {
            const totalItems = marketplaces.length + 1;
            const entryIndex = selectedIndex - 1;
            if (key.upArrow || input === "k") {
                setSelectedIndex((value) => Math.max(0, value - 1));
            } else if (key.downArrow || input === "j") {
                setSelectedIndex((value) => Math.min(totalItems - 1, value + 1));
            } else if (input === "u" || input === "U") {
                if (entryIndex >= 0) {
                    setMarketplaces((prev) =>
                        prev.map((entry, index) =>
                            index === entryIndex
                                ? {
                                    ...entry,
                                    pendingUpdate: !entry.pendingUpdate,
                                    pendingRemove: entry.pendingUpdate ? entry.pendingRemove : false
                                }
                                : entry
                        )
                    );
                }
            } else if (input === "r" || input === "R") {
                if (entryIndex >= 0) {
                    const entry = marketplaces[entryIndex];
                    if (entry) {
                        setSelectedMarketplace(entry);
                        setMode("confirm-remove");
                    }
                }
            } else if (key.return) {
                if (selectedIndex === 0) {
                    setViewState({ type: "add-marketplace" });
                } else if (hasPendingChanges()) {
                    void applyChanges();
                } else {
                    const entry = marketplaces[entryIndex];
                    if (entry) {
                        setSelectedMarketplace(entry);
                        setMode("details");
                        setDetailsIndex(0);
                    }
                }
            }
        } else if (mode === "details") {
            const options = buildOptions(selectedMarketplace);
            const maxIndex = options.length - 1;
            if (key.upArrow || input === "k") {
                setDetailsIndex((value) => Math.max(0, value - 1));
            } else if (key.downArrow || input === "j") {
                setDetailsIndex((value) => Math.min(maxIndex, value + 1));
            } else if (key.return && selectedMarketplace) {
                const selected = options[detailsIndex];
                if (selected?.value === "browse") {
                    setViewState({ type: "browse-marketplace", targetMarketplace: selectedMarketplace.name });
                } else if (selected?.value === "update") {
                    const updated = marketplaces.map((entry) =>
                        entry.name === selectedMarketplace.name ? { ...entry, pendingUpdate: true } : entry
                    );
                    setMarketplaces(updated);
                    void applyChanges(updated);
                } else if (selected?.value === "toggle-auto-update") {
                    void toggleAutoUpdate(selectedMarketplace);
                } else if (selected?.value === "remove") {
                    setMode("confirm-remove");
                }
            }
        } else if (mode === "confirm-remove") {
            if (input === "y" || input === "Y") {
                void confirmRemove();
            } else if (input === "n" || input === "N") {
                setMode("list");
                setSelectedMarketplace(null);
            }
        }
    });

    if (isLoading) {
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Text>Loading marketplaces…</Text>
                </Box>
            </Box>
        );
    }

    if (marketplaces.length === 0) {
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Box marginBottom={1}>
                        <Text bold>Manage marketplaces</Text>
                    </Box>
                    <Box flexDirection="row" gap={1}>
                        <Text color="claude">{figures.pointer} +</Text>
                        <Text bold color="claude">Add Marketplace</Text>
                    </Box>
                </Box>
                <Box marginLeft={3}>
                    <Text dimColor italic>
                        {exitState.pending
                            ? `Press ${exitState.keyName} again to go back`
                            : "Enter to select · Esc to go back"}
                    </Text>
                </Box>
            </Box>
        );
    }

    if (mode === "confirm-remove" && selectedMarketplace) {
        const installedCount = selectedMarketplace.installedPlugins?.length || 0;
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Text bold color="warning">
                        Remove marketplace <Text italic>{selectedMarketplace.name}</Text>?
                    </Text>
                    <Box flexDirection="column">
                        {installedCount > 0 && (
                            <Box marginTop={1}>
                                <Text color="warning">
                                    This will also uninstall {installedCount} plugin{installedCount !== 1 ? "s" : ""}{" "}
                                    from this marketplace:
                                </Text>
                            </Box>
                        )}
                        {selectedMarketplace.installedPlugins &&
                            selectedMarketplace.installedPlugins.length > 0 && (
                                <Box flexDirection="column" marginTop={1} marginLeft={2}>
                                    {selectedMarketplace.installedPlugins.map((plugin) => (
                                        <Text key={plugin.name} dimColor>
                                            • {plugin.name}
                                        </Text>
                                    ))}
                                </Box>
                            )}
                        <Box marginTop={1}>
                            <Text>
                                Press <Text bold>y</Text> to confirm or <Text bold>n</Text> to cancel
                            </Text>
                        </Box>
                    </Box>
                </Box>
            </Box>
        );
    }

    if (mode === "details" && selectedMarketplace) {
        const isUpdating = selectedMarketplace.pendingUpdate || isProcessing;
        const options = buildOptions(selectedMarketplace);

        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Text bold>{selectedMarketplace.name}</Text>
                    <Text dimColor>{selectedMarketplace.source}</Text>
                    <Box marginTop={1}>
                        <Text>{selectedMarketplace.pluginCount || 0} available plugin{selectedMarketplace.pluginCount !== 1 ? "s" : ""}</Text>
                    </Box>
                    {selectedMarketplace.installedPlugins && selectedMarketplace.installedPlugins.length > 0 && (
                        <Box flexDirection="column" marginTop={1}>
                            <Text bold>
                                Installed plugins ({selectedMarketplace.installedPlugins.length}):
                            </Text>
                            <Box flexDirection="column" marginLeft={1}>
                                {selectedMarketplace.installedPlugins.map((plugin) => (
                                    <Box key={plugin.name} flexDirection="row" gap={1}>
                                        <Text>{figures.bullet}</Text>
                                        <Box flexDirection="column">
                                            <Text>{plugin.name}</Text>
                                            <Text dimColor>{plugin.manifest.description}</Text>
                                        </Box>
                                    </Box>
                                ))}
                            </Box>
                        </Box>
                    )}
                    {isUpdating && (
                        <Box marginTop={1} flexDirection="column">
                            <Text color="claude">Updating marketplace…</Text>
                            {progressMessage && <Text dimColor>{progressMessage}</Text>}
                        </Box>
                    )}
                    {!isUpdating && successMessage && (
                        <Box marginTop={1}>
                            <Text color="claude">{successMessage}</Text>
                        </Box>
                    )}
                    {!isUpdating && errorMessage && (
                        <Box marginTop={1}>
                            <Text color="error">{errorMessage}</Text>
                        </Box>
                    )}
                    {!isUpdating && (
                        <Box flexDirection="column" marginTop={1}>
                            {options.map((option, index) => {
                                const isSelected = index === detailsIndex;
                                return (
                                    <Box key={option.value}>
                                        <Text color={isSelected ? "claude" : undefined}>
                                            {isSelected ? figures.pointer : " "} {option.label}
                                        </Text>
                                        {option.secondaryLabel && <Text dimColor> {option.secondaryLabel}</Text>}
                                    </Box>
                                );
                            })}
                        </Box>
                    )}
                    {!isUpdating && !isAutoUpdateLocked() && selectedMarketplace.autoUpdate && (
                        <Box marginTop={1}>
                            <Text dimColor>
                                Auto-update enabled. Claude Code will automatically update this marketplace and its installed plugins.
                            </Text>
                        </Box>
                    )}
                </Box>
                <Box marginLeft={3}>
                    <Text dimColor italic>
                        {isUpdating ? "Please wait…" : `${figures.arrowUp}${figures.arrowDown} · enter to select · Esc to go back`}
                    </Text>
                </Box>
            </Box>
        );
    }

    const { updateCount, removeCount } = pendingCounts();
    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                {children}
                <Box marginBottom={1}>
                    <Text bold>Manage marketplaces</Text>
                </Box>
                <Box flexDirection="row" gap={1} marginBottom={1}>
                    <Text color={selectedIndex === 0 ? "claude" : undefined}>
                        {selectedIndex === 0 ? figures.pointer : " "} +
                    </Text>
                    <Text bold color={selectedIndex === 0 ? "claude" : undefined}>
                        Add Marketplace
                    </Text>
                </Box>
                <Box flexDirection="column">
                    {marketplaces.map((entry, index) => {
                        const isSelected = index + 1 === selectedIndex;
                        const flags: string[] = [];
                        if (entry.pendingUpdate) flags.push("UPDATE");
                        if (entry.pendingRemove) flags.push("REMOVE");
                        const isOfficial = entry.name === "claude-plugins-official";
                        return (
                            <Box key={entry.name} flexDirection="row" gap={1} marginBottom={1}>
                                <Text color={isSelected ? "claude" : undefined}>
                                    {isSelected ? figures.pointer : " "} {entry.pendingRemove ? figures.cross : figures.bullet}
                                </Text>
                                <Box flexDirection="column" flexGrow={1}>
                                    <Box flexDirection="row" gap={1}>
                                        <Text bold strikethrough={entry.pendingRemove} dimColor={entry.pendingRemove}>
                                            {isOfficial && <Text color="claude">✻ </Text>}
                                            {entry.name}
                                            {isOfficial && <Text color="claude"> ✻</Text>}
                                        </Text>
                                        {flags.length > 0 && <Text color="warning">[{flags.join(", ")}]</Text>}
                                    </Box>
                                    <Text dimColor>{entry.source}</Text>
                                    <Text dimColor>
                                        {entry.pluginCount !== undefined && <>{entry.pluginCount} available</>}
                                        {entry.installedPlugins && entry.installedPlugins.length > 0 && (
                                            <> • {entry.installedPlugins.length} installed</>
                                        )}
                                        {entry.lastUpdated && <> • Updated {new Date(entry.lastUpdated).toLocaleDateString()}</>}
                                    </Text>
                                </Box>
                            </Box>
                        );
                    })}
                </Box>
                {hasPendingChanges() && (
                    <Box marginTop={1} flexDirection="column">
                        <Text>
                            <Text bold>Pending changes:</Text> <Text dimColor>Enter to apply</Text>
                        </Text>
                        {updateCount > 0 && (
                            <Text>• Update {updateCount} marketplace{updateCount > 1 ? "s" : ""}</Text>
                        )}
                        {removeCount > 0 && (
                            <Text color="warning">• Remove {removeCount} marketplace{removeCount > 1 ? "s" : ""}</Text>
                        )}
                    </Box>
                )}
                {isProcessing && (
                    <Box marginTop={1}>
                        <Text color="claude">Processing changes…</Text>
                    </Box>
                )}
                {errorMessage && (
                    <Box marginTop={1}>
                        <Text color="error">{errorMessage}</Text>
                    </Box>
                )}
            </Box>
            <MarketplaceFooter exitState={exitState} hasPendingActions={hasPendingChanges()} />
        </Box>
    );
}

export const MarketplaceManager = ManageMarketplaces;

export function usePagination({
    totalItems,
    maxVisible = 5,
    selectedIndex = 0
}: {
    totalItems: number;
    maxVisible?: number;
    selectedIndex?: number;
}) {
    const needsPagination = totalItems > maxVisible;
    const startRef = useRef(0);

    const startIndex = useMemo(() => {
        if (!needsPagination) return 0;
        const start = startRef.current;
        if (selectedIndex < start) {
            startRef.current = selectedIndex;
            return selectedIndex;
        }
        if (selectedIndex >= start + maxVisible) {
            const next = selectedIndex - maxVisible + 1;
            startRef.current = next;
            return next;
        }
        const maxStart = Math.max(0, totalItems - maxVisible);
        const clamped = Math.min(start, maxStart);
        startRef.current = clamped;
        return clamped;
    }, [selectedIndex, maxVisible, needsPagination, totalItems]);

    const endIndex = Math.min(startIndex + maxVisible, totalItems);

    const getVisibleItems = useCallback(
        <T,>(items: T[]) => {
            if (!needsPagination) return items;
            return items.slice(startIndex, endIndex);
        },
        [needsPagination, startIndex, endIndex]
    );

    const toActualIndex = useCallback((index: number) => startIndex + index, [startIndex]);
    const isOnCurrentPage = useCallback(
        (index: number) => index >= startIndex && index < endIndex,
        [startIndex, endIndex]
    );

    const goToPage = useCallback((_page: number) => { }, []);
    const nextPage = useCallback(() => { }, []);
    const prevPage = useCallback(() => { }, []);
    const handleSelectionChange = useCallback(
        (nextIndex: number, setIndex: (value: number) => void) => {
            const clamped = Math.max(0, Math.min(nextIndex, totalItems - 1));
            setIndex(clamped);
        },
        [totalItems]
    );
    const handlePageNavigation = useCallback((_direction: number, _setIndex: (value: number) => void) => {
        return false;
    }, []);

    const totalPages = Math.max(1, Math.ceil(totalItems / maxVisible));

    return {
        currentPage: Math.floor(startIndex / maxVisible),
        totalPages,
        startIndex,
        endIndex,
        needsPagination,
        pageSize: maxVisible,
        getVisibleItems,
        toActualIndex,
        isOnCurrentPage,
        goToPage,
        nextPage,
        prevPage,
        handleSelectionChange,
        handlePageNavigation,
        scrollPosition: {
            current: selectedIndex + 1,
            total: totalItems,
            canScrollUp: startIndex > 0,
            canScrollDown: startIndex + maxVisible < totalItems
        }
    };
}

export function getRepoNameFromSource(input: { entry: { source?: any } }) {
    if (
        input.entry.source &&
        typeof input.entry.source === "object" &&
        "source" in input.entry.source &&
        input.entry.source.source === "github" &&
        "repo" in input.entry.source
    ) {
        return input.entry.source.repo as string;
    }
    return null;
}

export function getPluginInstallOptions(hasHomepage: boolean, hasGitHub: boolean) {
    const options = [
        { label: "Install for you (user scope)", action: "install-user" },
        { label: "Install for all collaborators on this repository (project scope)", action: "install-project" },
        { label: "Install for you, in this repo only (local scope)", action: "install-local" }
    ];
    if (hasHomepage) options.push({ label: "Open homepage", action: "homepage" });
    if (hasGitHub) options.push({ label: "View on GitHub", action: "github" });
    options.push({ label: "Back to plugin list", action: "back" });
    return options;
}

export function initMarketplaceManager() { }

export function initPaginationHook() { }
