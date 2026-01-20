// Logic from chunk_842.ts / chunk_843.ts (Installed Plugins Manager)

import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import fs from "node:fs/promises";
import path from "node:path";
import { figures } from "../../vendor/terminalFigures.js";
import { usePagination } from "./MarketplaceManager.js";
import { PluginConfigurationForm, listPluginAgents } from "./PluginDiscoveryView.js";
import { openInBrowser } from "../../utils/browser/BrowserUtils.js";
import { loadMcpBundle } from "../../services/mcp/McpBundleLoader.js";
import { getSettings } from "../../services/terminal/settings.js";
import { InstalledPluginStore } from "../../services/plugin/InstalledPluginStore.js";
import { enablePlugin, disablePlugin, uninstallPlugin, updatePlugin } from "../../services/mcp/PluginManager.js";

type ViewState = { type: "menu" } | { type: "browse-marketplace"; targetMarketplace: string };

type InstalledPlugin = {
    name: string;
    path: string;
    source: string;
    manifest: {
        version?: string;
        description?: string;
        author?: { name?: string };
        homepage?: string;
        repository?: string;
        mcpServers?: string | string[] | Record<string, any>;
    };
    commandsPath?: string;
    commandsPaths?: string[];
    agentsPath?: string;
    agentsPaths?: string[];
    skillsPath?: string;
    skillsPaths?: string[];
    hooksConfig?: Record<string, any>;
    mcpServers?: Record<string, any>;
};

type MarketplaceGroup = {
    name: string;
    installedPlugins: InstalledPlugin[];
    enabledCount: number;
    disabledCount: number;
};

type PluginListEntry = {
    plugin: InstalledPlugin;
    marketplace: string;
    scope?: string;
    pendingEnable?: boolean;
    pendingUpdate: boolean;
};

type PluginConfigState = {
    manifest: { name: string };
    configSchema: Record<string, any>;
};

type InstalledPluginsViewProps = {
    setViewState: (state: ViewState) => void;
    setResult: (message: string) => void;
    onManageComplete?: () => Promise<void> | void;
    targetPlugin?: string;
    targetMarketplace?: string;
    action?: string;
    children?: React.ReactNode;
};

function trackError(_error: Error) { }

function isMcpBundleSource(source: string): boolean {
    return source.endsWith(".mcpb");
}

async function loadMarketplace(name: string): Promise<{ plugins: any[] }> {
    void name;
    return { plugins: [] };
}

async function getUpdateWarning(pluginName: string, marketplace: string): Promise<string | null> {
    const marketplaceData = await loadMarketplace(marketplace);
    const entry = marketplaceData?.plugins?.find((plugin) => plugin.name === pluginName);
    if (entry && typeof entry.source === "string") {
        return `Local plugins cannot be updated remotely. To update, modify the source at: ${entry.source}`;
    }
    return null;
}

async function listSkillDirectories(dirPath: string): Promise<string[]> {
    try {
        const entries = await fs.readdir(dirPath, { withFileTypes: true });
        const skillDirs: string[] = [];
        for (const entry of entries) {
            if (entry.isDirectory() || entry.isSymbolicLink()) {
                const skillPath = path.join(dirPath, entry.name, "SKILL.md");
                try {
                    await fs.access(skillPath);
                    skillDirs.push(entry.name);
                } catch {
                    // Skip directories without a SKILL.md file.
                }
            }
        }
        return skillDirs;
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        trackError(err instanceof Error ? err : Error(`Failed to read skill directories: ${message}`));
        return [];
    }
}

function getEnabledPluginsMap() {
    const settings = getSettings("userSettings") as { enabledPlugins?: Record<string, boolean> };
    return settings?.enabledPlugins ?? {};
}

function resolvePluginScope(pluginId: string) {
    const store = InstalledPluginStore.getAllInstalledPlugins();
    const installs = store?.plugins?.[pluginId];
    if (Array.isArray(installs) && installs.length > 0) {
        return { scope: installs[0].scope as string | undefined };
    }
    return { scope: undefined };
}

function canMutateScope(scope?: string) {
    return scope !== "managed";
}

function isPluginInstalled(pluginId: string) {
    const store = InstalledPluginStore.getAllInstalledPlugins();
    return Boolean(store?.plugins?.[pluginId]);
}

async function installMissingPlugin(pluginId: string, scope: string) {
    const result = await enablePlugin(pluginId, scope);
    if (!result.success) {
        throw new Error(result.message);
    }
}

async function configurePluginBundle(
    bundlePath: string,
    pluginPath: string,
    pluginId: string,
    config?: Record<string, any>
): Promise<any> {
    return loadMcpBundle(bundlePath, pluginPath, pluginId, undefined, config ?? {});
}

function PluginComponentsView({ plugin, marketplace }: { plugin: InstalledPlugin; marketplace: string }) {
    const [components, setComponents] = useState<{
        commands: string[] | null;
        agents: string[] | null;
        skills: string[] | null;
        hooks: string[] | null;
        mcpServers: string[] | null;
    } | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function loadComponents() {
            try {
                const marketplaceData = await loadMarketplace(marketplace);
                const entry = marketplaceData?.plugins?.find((candidate) => candidate.name === plugin.name);
                if (!entry) {
                    setError(`Plugin ${plugin.name} not found in marketplace`);
                    return;
                }

                const commandPaths: string[] = [];
                if (plugin.commandsPath) commandPaths.push(plugin.commandsPath);
                if (plugin.commandsPaths) commandPaths.push(...plugin.commandsPaths);

                const commands: string[] = [];
                for (const commandPath of commandPaths) {
                    if (typeof commandPath === "string") {
                        const found = await listPluginAgents(commandPath);
                        commands.push(...found);
                    }
                }

                const agentPaths: string[] = [];
                if (plugin.agentsPath) agentPaths.push(plugin.agentsPath);
                if (plugin.agentsPaths) agentPaths.push(...plugin.agentsPaths);

                const agents: string[] = [];
                for (const agentPath of agentPaths) {
                    if (typeof agentPath === "string") {
                        const found = await listPluginAgents(agentPath);
                        agents.push(...found);
                    }
                }

                const skillPaths: string[] = [];
                if (plugin.skillsPath) skillPaths.push(plugin.skillsPath);
                if (plugin.skillsPaths) skillPaths.push(...plugin.skillsPaths);

                const skills: string[] = [];
                for (const skillPath of skillPaths) {
                    if (typeof skillPath === "string") {
                        const found = await listSkillDirectories(skillPath);
                        skills.push(...found);
                    }
                }

                const hooks: string[] = [];
                if (plugin.hooksConfig) hooks.push(...Object.keys(plugin.hooksConfig));
                if (entry.hooks) hooks.push(...entry.hooks);

                const mcpServers: string[] = [];
                if (plugin.mcpServers) mcpServers.push(...Object.keys(plugin.mcpServers));
                if (entry.mcpServers) mcpServers.push(...entry.mcpServers);

                setComponents({
                    commands: commands.length > 0 ? commands : null,
                    agents: agents.length > 0 ? agents : null,
                    skills: skills.length > 0 ? skills : null,
                    hooks: hooks.length > 0 ? hooks : null,
                    mcpServers: mcpServers.length > 0 ? mcpServers : null
                });
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to load components");
            } finally {
                setLoading(false);
            }
        }

        loadComponents();
    }, [plugin.name, plugin.commandsPath, plugin.commandsPaths, plugin.agentsPath, plugin.agentsPaths, plugin.skillsPath, plugin.skillsPaths, plugin.hooksConfig, plugin.mcpServers, marketplace]);

    if (loading || !components) return null;
    if (error) {
        return (
            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Components:</Text>
                <Text dimColor>Error: {error}</Text>
            </Box>
        );
    }

    if (!(components.commands || components.agents || components.skills || components.hooks || components.mcpServers)) {
        return null;
    }

    const formatList = (value: string[] | null) => (value ? value.join(", ") : "");

    return (
        <Box flexDirection="column" marginBottom={1}>
            <Text bold>Installed components:</Text>
            {components.commands && <Text dimColor>• Commands: {formatList(components.commands)}</Text>}
            {components.agents && <Text dimColor>• Agents: {formatList(components.agents)}</Text>}
            {components.skills && <Text dimColor>• Skills: {formatList(components.skills)}</Text>}
            {components.hooks && <Text dimColor>• Hooks: {formatList(components.hooks)}</Text>}
            {components.mcpServers && <Text dimColor>• MCP Servers: {formatList(components.mcpServers)}</Text>}
        </Box>
    );
}

export function InstalledPluginsView({
    setViewState,
    setResult,
    onManageComplete,
    targetPlugin,
    targetMarketplace,
    children
}: InstalledPluginsViewProps) {
    const [view, setView] = useState<"plugin-list" | "plugin-details" | "configuring">("plugin-list");
    const [selected, setSelected] = useState<PluginListEntry | null>(null);
    const [marketplaces, setMarketplaces] = useState<MarketplaceGroup[]>([]);
    const [plugins, setPlugins] = useState<PluginListEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [detailsIndex, setDetailsIndex] = useState(0);
    const [isProcessing, setIsProcessing] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [configState, setConfigState] = useState<PluginConfigState | null>(null);
    const [configLoading, setConfigLoading] = useState(false);
    const [hasConfigurableBundle, setHasConfigurableBundle] = useState(false);

    const hasPendingChanges = plugins.some(
        (entry) => entry.pendingEnable !== undefined || entry.pendingUpdate
    );

    const pagination = usePagination({
        totalItems: plugins.length + (hasPendingChanges ? 1 : 0),
        selectedIndex
    });

    useEffect(() => {
        if (!selected) {
            setHasConfigurableBundle(false);
            return;
        }

        async function detectConfigurableBundle() {
            const mcpServers = selected?.plugin.manifest.mcpServers;
            let hasBundle =
                (typeof mcpServers === "string" && isMcpBundleSource(mcpServers)) ||
                (Array.isArray(mcpServers) && mcpServers.some((entry) => typeof entry === "string" && isMcpBundleSource(entry)));

            if (!hasBundle && selected) {
                try {
                    const parentDir = path.join(selected.plugin.path, "..");
                    const marketplacePath = path.join(parentDir, ".claude-plugin", "marketplace.json");
                    const fileContents = await fs.readFile(marketplacePath, "utf-8");
                    const manifest = JSON.parse(fileContents).plugins?.find((entry: any) => entry.name === selected.plugin.name);
                    const bundle = manifest?.mcpServers;
                    if (bundle) {
                        hasBundle =
                            (typeof bundle === "string" && isMcpBundleSource(bundle)) ||
                            (Array.isArray(bundle) && bundle.some((entry: any) => typeof entry === "string" && isMcpBundleSource(entry)));
                    }
                } catch (err) {
                    console.log(`Failed to read raw marketplace.json: ${err}`);
                }
            }

            setHasConfigurableBundle(hasBundle);
        }

        void detectConfigurableBundle();
    }, [selected]);

    useEffect(() => {
        async function loadInstalledPlugins() {
            setLoading(true);
            try {
                const store = InstalledPluginStore.getAllInstalledPlugins();
                const entries = Object.entries(store.plugins ?? {});
                const settings = getEnabledPluginsMap();
                const grouped: Record<string, InstalledPlugin[]> = {};

                for (const [pluginId, installs] of entries) {
                    const [name, marketplace = "local"] = pluginId.split("@");
                    const install = Array.isArray(installs) && installs.length > 0 ? installs[0] : null;
                    const plugin: InstalledPlugin = {
                        name,
                        path: install?.installPath ?? "",
                        source: pluginId,
                        manifest: {
                            version: install?.version
                        }
                    };
                    if (!grouped[marketplace]) grouped[marketplace] = [];
                    grouped[marketplace].push(plugin);
                }

                const groupList: MarketplaceGroup[] = [];
                for (const [marketplace, installedPlugins] of Object.entries(grouped)) {
                    const enabledCount = installedPlugins.filter((plugin) => {
                        const pluginId = `${plugin.name}@${marketplace}`;
                        return settings?.[pluginId] !== false;
                    }).length;
                    groupList.push({
                        name: marketplace,
                        installedPlugins,
                        enabledCount,
                        disabledCount: installedPlugins.length - enabledCount
                    });
                }

                groupList.sort((a, b) => {
                    if (a.name === "claude-plugin-directory") return -1;
                    if (b.name === "claude-plugin-directory") return 1;
                    return a.name.localeCompare(b.name);
                });

                setMarketplaces(groupList);

                const listEntries: PluginListEntry[] = [];
                for (const group of groupList) {
                    for (const plugin of group.installedPlugins) {
                        const pluginId = `${plugin.name}@${group.name}`;
                        const { scope } = resolvePluginScope(pluginId);
                        listEntries.push({
                            plugin,
                            marketplace: group.name,
                            scope,
                            pendingEnable: undefined,
                            pendingUpdate: false
                        });
                    }
                }

                setPlugins(listEntries);
                setSelectedIndex(0);
            } finally {
                setLoading(false);
            }
        }

        void loadInstalledPlugins();
    }, []);

    useEffect(() => {
        if (targetPlugin && marketplaces.length > 0 && !loading) {
            const groupCandidates = targetMarketplace
                ? marketplaces.filter((group) => group.name === targetMarketplace)
                : marketplaces;

            for (const group of groupCandidates) {
                const match = group.installedPlugins.find((plugin) => plugin.name === targetPlugin);
                if (match) {
                    const pluginId = `${match.name}@${group.name}`;
                    const { scope } = resolvePluginScope(pluginId);
                    setSelected({
                        plugin: match,
                        marketplace: group.name,
                        scope,
                        pendingEnable: undefined,
                        pendingUpdate: false
                    });
                    setView("plugin-details");
                    break;
                }
            }
        }
    }, [targetPlugin, targetMarketplace, marketplaces, loading]);

    const pendingCounts = useMemo(() => {
        const updateCount = plugins.filter((entry) => entry.pendingUpdate).length;
        const enableCount = plugins.filter((entry) => entry.pendingEnable === true).length;
        const disableCount = plugins.filter((entry) => entry.pendingEnable === false).length;
        return { updateCount, enableCount, disableCount };
    }, [plugins]);

    const applyPendingChanges = async () => {
        setIsProcessing(true);
        setErrorMessage(null);
        try {
            let updated = 0;
            let enabled = 0;
            let disabled = 0;

            for (const entry of plugins) {
                const pluginId = `${entry.plugin.name}@${entry.marketplace}`;
                const scope = entry.scope || "user";

                if (entry.pendingUpdate) {
                    const result = await updatePlugin(pluginId, scope);
                    if (result.success && !result.alreadyUpToDate) updated++;
                }

                if (entry.pendingEnable !== undefined && canMutateScope(scope)) {
                    if (entry.pendingEnable) {
                        if (!isPluginInstalled(pluginId)) {
                            await installMissingPlugin(pluginId, scope);
                        }
                        const result = await enablePlugin(pluginId, scope);
                        if (result.success) enabled++;
                    } else {
                        const result = await disablePlugin(pluginId, scope);
                        if (result.success) disabled++;
                    }
                }
            }

            const summaryParts: string[] = [];
            if (updated > 0) summaryParts.push(`Updated ${updated} plugin${updated !== 1 ? "s" : ""}`);
            if (enabled > 0) summaryParts.push(`Enabled ${enabled} plugin${enabled !== 1 ? "s" : ""}`);
            if (disabled > 0) summaryParts.push(`Disabled ${disabled} plugin${disabled !== 1 ? "s" : ""}`);

            if (summaryParts.length > 0) {
                setResult(`✓ ${summaryParts.join(", ")}. Restart Claude Code to apply changes.`);
            }

            if (onManageComplete) await onManageComplete();
            setViewState({ type: "menu" });
        } catch (err) {
            setIsProcessing(false);
            const message = err instanceof Error ? err.message : String(err);
            setErrorMessage(`Failed to apply changes: ${message}`);
            trackError(err instanceof Error ? err : Error(`Failed to apply plugin changes: ${message}`));
        }
    };

    const applyPluginAction = async (action: "enable" | "disable" | "uninstall" | "update") => {
        if (!selected) return;
        const scope = selected.scope || "user";
        if (!canMutateScope(scope) && action !== "update") {
            setErrorMessage("Managed plugins can only be updated, not enabled, disabled, or uninstalled.");
            return;
        }

        setIsProcessing(true);
        setErrorMessage(null);
        try {
            const pluginId = `${selected.plugin.name}@${selected.marketplace}`;
            switch (action) {
                case "enable": {
                    if (!canMutateScope(scope)) break;
                    if (!isPluginInstalled(pluginId)) {
                        await installMissingPlugin(pluginId, scope);
                    }
                    const result = await enablePlugin(pluginId, scope);
                    if (!result.success) throw new Error(result.message);
                    break;
                }
                case "disable": {
                    if (!canMutateScope(scope)) break;
                    const result = await disablePlugin(pluginId, scope);
                    if (!result.success) throw new Error(result.message);
                    break;
                }
                case "uninstall": {
                    if (!canMutateScope(scope)) break;
                    const result = await uninstallPlugin(pluginId, scope);
                    if (!result.success) throw new Error(result.message);
                    break;
                }
                case "update": {
                    const result = await updatePlugin(pluginId, scope);
                    if (!result.success) throw new Error(result.message);
                    if (result.alreadyUpToDate) {
                        setResult(`${selected.plugin.name} is already at the latest version (${result.newVersion}).`);
                        if (onManageComplete) await onManageComplete();
                        setViewState({ type: "menu" });
                        return;
                    }
                    break;
                }
            }

            const verb =
                action === "enable" ? "Enabled" : action === "disable" ? "Disabled" : action === "update" ? "Updated" : "Uninstalled";
            setResult(`✓ ${verb} ${selected.plugin.name}. Restart Claude Code to apply changes.`);
            if (onManageComplete) await onManageComplete();
            setViewState({ type: "menu" });
        } catch (err) {
            setIsProcessing(false);
            const message = err instanceof Error ? err.message : String(err);
            setErrorMessage(`Failed to ${action}: ${message}`);
            trackError(err instanceof Error ? err : Error(`Failed to ${action} plugin: ${message}`));
        }
    };

    const uninstallFromList = async (entry: PluginListEntry) => {
        const scope = entry.scope || "user";
        if (!canMutateScope(scope)) {
            setErrorMessage("Managed plugins cannot be uninstalled. They can only be updated.");
            return;
        }
        setIsProcessing(true);
        setErrorMessage(null);
        try {
            const pluginId = `${entry.plugin.name}@${entry.marketplace}`;
            const result = await uninstallPlugin(pluginId, scope);
            if (!result.success) throw new Error(result.message);

            const store = InstalledPluginStore.getAllInstalledPlugins();
            const entries = Object.entries(store.plugins ?? {});
            const settings = getEnabledPluginsMap();
            const grouped: Record<string, InstalledPlugin[]> = {};

            for (const [id, installs] of entries) {
                const [name, marketplace = "local"] = id.split("@");
                const install = Array.isArray(installs) && installs.length > 0 ? installs[0] : null;
                const plugin: InstalledPlugin = {
                    name,
                    path: install?.installPath ?? "",
                    source: id,
                    manifest: {
                        version: install?.version
                    }
                };
                if (!grouped[marketplace]) grouped[marketplace] = [];
                grouped[marketplace].push(plugin);
            }

            const groupList: MarketplaceGroup[] = [];
            for (const [marketplace, installedPlugins] of Object.entries(grouped)) {
                const enabledCount = installedPlugins.filter((plugin) => {
                    const pluginIdLocal = `${plugin.name}@${marketplace}`;
                    return settings?.[pluginIdLocal] !== false;
                }).length;
                groupList.push({
                    name: marketplace,
                    installedPlugins,
                    enabledCount,
                    disabledCount: installedPlugins.length - enabledCount
                });
            }

            groupList.sort((a, b) => {
                if (a.name === "claude-plugin-directory") return -1;
                if (b.name === "claude-plugin-directory") return 1;
                return a.name.localeCompare(b.name);
            });

            setMarketplaces(groupList);
            const listEntries: PluginListEntry[] = [];
            for (const group of groupList) {
                for (const plugin of group.installedPlugins) {
                    const pluginIdLocal = `${plugin.name}@${group.name}`;
                    const { scope: resolvedScope } = resolvePluginScope(pluginIdLocal);
                    listEntries.push({
                        plugin,
                        marketplace: group.name,
                        scope: resolvedScope,
                        pendingEnable: undefined,
                        pendingUpdate: false
                    });
                }
            }

            setPlugins(listEntries);
            if (selectedIndex >= listEntries.length) {
                setSelectedIndex(Math.max(0, listEntries.length - 1));
            }
            setResult(`✓ Uninstalled ${entry.plugin.name}. Restart Claude Code to apply changes.`);
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            setErrorMessage(`Failed to uninstall: ${message}`);
            trackError(err instanceof Error ? err : Error(`Failed to uninstall plugin: ${message}`));
        } finally {
            setIsProcessing(false);
        }
    };

    useInput((input, key) => {
        if (key.escape) {
            if (view === "plugin-details") {
                setView("plugin-list");
                setSelected(null);
                setErrorMessage(null);
            } else if (view === "configuring") {
                setView("plugin-details");
                setConfigState(null);
            } else {
                setViewState({ type: "menu" });
            }
            return;
        }

        if (view === "plugin-list") {
            const totalItems = plugins.length + (hasPendingChanges ? 1 : 0);
            if ((key.upArrow || input === "k") && selectedIndex > 0) {
                pagination.handleSelectionChange(selectedIndex - 1, setSelectedIndex);
            } else if ((key.downArrow || input === "j") && selectedIndex < totalItems - 1) {
                pagination.handleSelectionChange(selectedIndex + 1, setSelectedIndex);
            } else if (input === " " && selectedIndex < plugins.length) {
                const next = [...plugins];
                const entry = next[selectedIndex];
                if (entry) {
                    const settings = getEnabledPluginsMap();
                    const pluginId = `${entry.plugin.name}@${entry.marketplace}`;
                    const isEnabled = settings?.[pluginId] !== false;
                    if (entry.pendingEnable === undefined) {
                        entry.pendingEnable = !isEnabled;
                    } else {
                        entry.pendingEnable = undefined;
                    }
                    setPlugins(next);
                }
            } else if (input === "u" && selectedIndex < plugins.length) {
                const next = [...plugins];
                const entry = next[selectedIndex];
                if (entry) {
                    void (async () => {
                        try {
                            const warning = await getUpdateWarning(entry.plugin.name, entry.marketplace);
                            if (warning) {
                                setErrorMessage(warning);
                                return;
                            }
                            entry.pendingUpdate = !entry.pendingUpdate;
                            setPlugins(next);
                        } catch (err) {
                            setErrorMessage(
                                err instanceof Error ? err.message : "Failed to check plugin update availability"
                            );
                        }
                    })();
                }
            } else if ((key.delete || key.backspace) && selectedIndex < plugins.length && !isProcessing) {
                const entry = plugins[selectedIndex];
                if (entry) {
                    void uninstallFromList(entry);
                }
            } else if (key.return) {
                if (selectedIndex === plugins.length && hasPendingChanges) {
                    void applyPendingChanges();
                } else if (selectedIndex < plugins.length) {
                    const entry = plugins[selectedIndex];
                    if (entry) {
                        setSelected(entry);
                        setView("plugin-details");
                        setDetailsIndex(0);
                        setErrorMessage(null);
                    }
                }
            }
        } else if (view === "plugin-details" && selected) {
            const settings = getEnabledPluginsMap();
            const pluginId = `${selected.plugin.name}@${selected.marketplace}`;
            const isEnabled = settings?.[pluginId] !== false;
            const options = [
                {
                    label: isEnabled ? "Disable plugin" : "Enable plugin",
                    action: async () => { await applyPluginAction(isEnabled ? "disable" : "enable"); }
                },
                {
                    label: selected.pendingUpdate ? "Unmark for update" : "Mark for update",
                    action: async () => {
                        try {
                            const warning = await getUpdateWarning(selected.plugin.name, selected.marketplace);
                            if (warning) {
                                setErrorMessage(warning);
                                return;
                            }
                            const next = [...plugins];
                            const index = next.findIndex(
                                (entry) =>
                                    entry.plugin.name === selected.plugin.name &&
                                    entry.marketplace === selected.marketplace
                            );
                            if (index !== -1) {
                                next[index].pendingUpdate = !selected.pendingUpdate;
                                setPlugins(next);
                                setSelected({
                                    ...selected,
                                    pendingUpdate: !selected.pendingUpdate
                                });
                            }
                        } catch (err) {
                            setErrorMessage(
                                err instanceof Error ? err.message : "Failed to check plugin update availability"
                            );
                        }
                    }
                }
            ];

            if (hasConfigurableBundle) {
                options.push({
                    label: "Configure",
                    action: async () => {
                        setConfigLoading(true);
                        try {
                            const bundles = selected.plugin.manifest.mcpServers;
                            let bundlePath: string | null = null;
                            if (typeof bundles === "string" && isMcpBundleSource(bundles)) {
                                bundlePath = bundles;
                            } else if (Array.isArray(bundles)) {
                                for (const bundle of bundles) {
                                    if (typeof bundle === "string" && isMcpBundleSource(bundle)) {
                                        bundlePath = bundle;
                                        break;
                                    }
                                }
                            }
                            if (!bundlePath) {
                                setErrorMessage("No MCPB file found in plugin");
                                setConfigLoading(false);
                                return;
                            }

                            const result = await configurePluginBundle(
                                bundlePath,
                                selected.plugin.path,
                                pluginId,
                                undefined
                            );
                            if ("status" in result && result.status === "needs-config") {
                                setConfigState(result as PluginConfigState);
                                setView("configuring");
                            } else {
                                setErrorMessage("Failed to load MCPB for configuration");
                            }
                        } catch (err) {
                            const message = err instanceof Error ? err.message : String(err);
                            setErrorMessage(`Failed to load configuration: ${message}`);
                        } finally {
                            setConfigLoading(false);
                        }
                    }
                });
            }

            options.push({
                label: "Update now",
                action: async () => { await applyPluginAction("update"); }
            });
            options.push({
                label: "Uninstall",
                action: async () => { await applyPluginAction("uninstall"); }
            });
            if (selected.plugin.manifest.homepage) {
                options.push({
                    label: "Open homepage",
                    action: async () => { await openInBrowser(selected.plugin.manifest.homepage as string); }
                });
            }
            if (selected.plugin.manifest.repository) {
                options.push({
                    label: "View on GitHub",
                    action: async () => { await openInBrowser(selected.plugin.manifest.repository as string); }
                });
            }
            options.push({
                label: "Back to plugin list",
                action: async () => {
                    setView("plugin-list");
                    setSelected(null);
                    setErrorMessage(null);
                }
            });

            if ((key.upArrow || input === "k") && detailsIndex > 0) {
                setDetailsIndex(detailsIndex - 1);
            } else if ((key.downArrow || input === "j") && detailsIndex < options.length - 1) {
                setDetailsIndex(detailsIndex + 1);
            } else if (key.return && options[detailsIndex]) {
                options[detailsIndex].action();
            }
        }
    });

    if (loading) {
        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Text>Loading installed plugins…</Text>
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
                        <Text bold>Manage plugins</Text>
                    </Box>
                    <Text>No plugins installed.</Text>
                </Box>
                <Box marginTop={1} paddingLeft={1}>
                    <Text dimColor>Esc to go back</Text>
                </Box>
            </Box>
        );
    }

    if (view === "configuring" && configState && selected) {
        const pluginId = `${selected.plugin.name}@${selected.marketplace}`;
        const handleCancel = () => {
            setConfigState(null);
            setView("plugin-details");
        };

        const handleSave = async (values: Record<string, any>) => {
            if (!configState || !selected) return;
            try {
                const mcpServers = selected.plugin.manifest.mcpServers;
                let bundlePath: string | null = null;
                if (typeof mcpServers === "string" && isMcpBundleSource(mcpServers)) bundlePath = mcpServers;
                else if (Array.isArray(mcpServers)) {
                    for (const entry of mcpServers) {
                        if (typeof entry === "string" && isMcpBundleSource(entry)) {
                            bundlePath = entry;
                            break;
                        }
                    }
                }
                if (!bundlePath) {
                    setErrorMessage("No MCPB file found");
                    setView("plugin-details");
                    return;
                }
                await configurePluginBundle(bundlePath, selected.plugin.path, pluginId, values);
                setErrorMessage(null);
                setConfigState(null);
                setView("plugin-details");
                setResult("Configuration saved. Restart Claude Code for changes to take effect.");
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                setErrorMessage(`Failed to save configuration: ${message}`);
                setView("plugin-details");
            }
        };

        return (
            <PluginConfigurationForm
                pluginName={selected.plugin.name}
                serverName={configState.manifest.name}
                configSchema={configState.configSchema}
                onSave={handleSave}
                onCancel={handleCancel}
            />
        );
    }

    if (view === "plugin-details" && selected) {
        const settings = getEnabledPluginsMap();
        const pluginId = `${selected.plugin.name}@${selected.marketplace}`;
        const isEnabled = settings?.[pluginId] !== false;
        const options = [
            {
                label: isEnabled ? "Disable plugin" : "Enable plugin",
                action: async () => { await applyPluginAction(isEnabled ? "disable" : "enable"); }
            },
            {
                label: selected.pendingUpdate ? "Unmark for update" : "Mark for update",
                action: async () => {
                    try {
                        const warning = await getUpdateWarning(selected.plugin.name, selected.marketplace);
                        if (warning) {
                            setErrorMessage(warning);
                            return;
                        }
                        const next = [...plugins];
                        const index = next.findIndex(
                            (entry) =>
                                entry.plugin.name === selected.plugin.name &&
                                entry.marketplace === selected.marketplace
                        );
                        if (index !== -1) {
                            next[index].pendingUpdate = !selected.pendingUpdate;
                            setPlugins(next);
                            setSelected({
                                ...selected,
                                pendingUpdate: !selected.pendingUpdate
                            });
                        }
                    } catch (err) {
                        setErrorMessage(
                            err instanceof Error ? err.message : "Failed to check plugin update availability"
                        );
                    }
                }
            }
        ];

        if (hasConfigurableBundle) {
            options.push({
                label: "Configure",
                action: async () => {
                    setConfigLoading(true);
                    try {
                        const bundles = selected.plugin.manifest.mcpServers;
                        let bundlePath: string | null = null;
                        if (typeof bundles === "string" && isMcpBundleSource(bundles)) {
                            bundlePath = bundles;
                        } else if (Array.isArray(bundles)) {
                            for (const bundle of bundles) {
                                if (typeof bundle === "string" && isMcpBundleSource(bundle)) {
                                    bundlePath = bundle;
                                    break;
                                }
                            }
                        }
                        if (!bundlePath) {
                            setErrorMessage("No MCPB file found in plugin");
                            setConfigLoading(false);
                            return;
                        }

                        const result = await configurePluginBundle(bundlePath, selected.plugin.path, pluginId, undefined);
                        if ("status" in result && result.status === "needs-config") {
                            setConfigState(result as PluginConfigState);
                            setView("configuring");
                        }
                    } catch (err) {
                        const message = err instanceof Error ? err.message : String(err);
                        setErrorMessage(`Failed to load configuration: ${message}`);
                    } finally {
                        setConfigLoading(false);
                    }
                }
            });
        }

        options.push({
            label: "Update now",
            action: async () => { await applyPluginAction("update"); }
        });
        options.push({
            label: "Uninstall",
            action: async () => { await applyPluginAction("uninstall"); }
        });
        if (selected.plugin.manifest.homepage) {
            options.push({
                label: "Open homepage",
                action: async () => { await openInBrowser(selected.plugin.manifest.homepage as string); }
            });
        }
        if (selected.plugin.manifest.repository) {
            options.push({
                label: "View on GitHub",
                action: async () => { await openInBrowser(selected.plugin.manifest.repository as string); }
            });
        }
        options.push({
            label: "Back to plugin list",
            action: async () => {
                setView("plugin-list");
                setSelected(null);
                setErrorMessage(null);
            }
        });

        return (
            <Box flexDirection="column">
                <Box flexDirection="column" paddingX={1} borderStyle="round">
                    {children}
                    <Box marginBottom={1}>
                        <Text bold>
                            {selected.plugin.name} @ {selected.marketplace}
                        </Text>
                    </Box>
                    <Box marginBottom={1}>
                        <Text dimColor>Scope: </Text>
                        <Text>{selected.scope || "user"}</Text>
                    </Box>
                    {selected.plugin.manifest.version && (
                        <Box marginBottom={1}>
                            <Text dimColor>Version: </Text>
                            <Text>{selected.plugin.manifest.version}</Text>
                        </Box>
                    )}
                    {selected.plugin.manifest.description && (
                        <Box marginBottom={1}>
                            <Text>{selected.plugin.manifest.description}</Text>
                        </Box>
                    )}
                    {selected.plugin.manifest.author?.name && (
                        <Box marginBottom={1}>
                            <Text dimColor>Author: </Text>
                            <Text>{selected.plugin.manifest.author.name}</Text>
                        </Box>
                    )}
                    <Box marginBottom={1}>
                        <Text dimColor>Status: </Text>
                        <Text color={isEnabled ? "success" : "warning"}>
                            {isEnabled ? "Enabled" : "Disabled"}
                        </Text>
                        {selected.pendingUpdate && <Text color="suggestion"> · Marked for update</Text>}
                    </Box>
                    <PluginComponentsView plugin={selected.plugin} marketplace={selected.marketplace} />
                    <Box marginTop={1} flexDirection="column">
                        {options.map((option, index) => {
                            const isSelected = index === detailsIndex;
                            return (
                                <Box key={option.label}>
                                    <Text color={isSelected ? "suggestion" : undefined}>
                                        {isSelected ? figures.pointer : " "}{" "}
                                    </Text>
                                    <Text
                                        bold={isSelected}
                                        color={
                                            option.label.includes("Uninstall")
                                                ? "error"
                                                : option.label.includes("Update")
                                                    ? "suggestion"
                                                    : undefined
                                        }
                                    >
                                        {option.label}
                                    </Text>
                                </Box>
                            );
                        })}
                    </Box>
                    {(isProcessing || configLoading) && (
                        <Box marginTop={1}>
                            <Text>Processing…</Text>
                        </Box>
                    )}
                    {errorMessage && (
                        <Box marginTop={1}>
                            <Text color="error">{errorMessage}</Text>
                        </Box>
                    )}
                </Box>
                <Box marginTop={1} paddingLeft={1}>
                    <Text dimColor>
                        <Text bold>Navigate:</Text> {figures.arrowUp}
                        {figures.arrowDown} • <Text bold>Select:</Text> Enter • <Text bold>Back:</Text> Esc
                    </Text>
                </Box>
            </Box>
        );
    }

    const visibleItems = pagination.getVisibleItems(plugins);

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                {children}
                <Box marginBottom={1}>
                    <Text bold>Installed Plugins</Text>
                    {pagination.needsPagination && (
                        <Text dimColor>
                            {" "}({pagination.scrollPosition.current}/{pagination.scrollPosition.total})
                        </Text>
                    )}
                </Box>
                {pagination.scrollPosition.canScrollUp && (
                    <Box>
                        <Text dimColor> {figures.arrowUp} more above</Text>
                    </Box>
                )}
                {visibleItems.map((entry, index) => {
                    const actualIndex = pagination.toActualIndex(index);
                    const settings = getEnabledPluginsMap();
                    const pluginId = `${entry.plugin.name}@${entry.marketplace}`;
                    const isEnabled = settings?.[pluginId] !== false;
                    const displayEnabled = entry.pendingEnable !== undefined ? entry.pendingEnable : isEnabled;
                    const isSelected = actualIndex === selectedIndex;
                    const isPending = entry.pendingEnable !== undefined || entry.pendingUpdate;
                    const previousEntry = index > 0 ? visibleItems[index - 1] : null;
                    const showMarketplaceHeader = !previousEntry || previousEntry.marketplace !== entry.marketplace;

                    return (
                        <Box key={pluginId} flexDirection="column">
                            {showMarketplaceHeader && (
                                <Box marginTop={index > 0 ? 1 : 0} marginBottom={0}>
                                    <Text dimColor bold>
                                        {entry.marketplace}
                                    </Text>
                                </Box>
                            )}
                            <Box>
                                <Text color={isSelected ? "suggestion" : undefined}>
                                    {isSelected ? figures.pointer : " "}{" "}
                                </Text>
                                <Text color={entry.pendingEnable !== undefined ? "warning" : displayEnabled ? "success" : undefined}>
                                    {displayEnabled ? figures.radioOn : figures.radioOff}{" "}
                                </Text>
                                <Text
                                    bold={isSelected}
                                    color={entry.pendingUpdate ? "suggestion" : isPending ? "warning" : undefined}
                                >
                                    {entry.plugin.name}
                                </Text>
                                {entry.scope && <Text dimColor> {entry.scope}</Text>}
                                {entry.plugin.manifest.version && <Text dimColor>, v{entry.plugin.manifest.version}</Text>}
                                {entry.pendingUpdate && <Text color="suggestion"> · update</Text>}
                            </Box>
                        </Box>
                    );
                })}
                {pagination.scrollPosition.canScrollDown && (
                    <Box>
                        <Text dimColor> {figures.arrowDown} more below</Text>
                    </Box>
                )}
                {hasPendingChanges && (
                    <Box marginTop={1}>
                        <Text>{selectedIndex === plugins.length ? figures.pointer : " "}</Text>
                        <Text bold={selectedIndex === plugins.length} color="success">
                            Apply changes
                        </Text>
                        <Text dimColor>
                            {" "}
                            {pendingCounts.updateCount > 0 && `(update ${pendingCounts.updateCount})`}
                            {pendingCounts.enableCount > 0 && ` (enable ${pendingCounts.enableCount})`}
                            {pendingCounts.disableCount > 0 && ` (disable ${pendingCounts.disableCount})`}
                        </Text>
                    </Box>
                )}
                {hasPendingChanges && (
                    <Box marginTop={1} paddingLeft={1}>
                        <Text color="warning">Restart to apply changes</Text>
                    </Box>
                )}
            </Box>
            <Box paddingLeft={3}>
                <Text dimColor italic>Space: toggle · Enter: details · Delete: uninstall · Esc: back</Text>
            </Box>
        </Box>
    );
}

export function initInstalledPluginsView() { }
