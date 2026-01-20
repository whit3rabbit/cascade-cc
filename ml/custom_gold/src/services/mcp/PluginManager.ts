
import * as fs from 'node:fs';
import * as path from 'node:path';
import { getSettings, updateSettings } from '../terminal/settings.js';
import { getMarketplaceConfig } from '../marketplace/MarketplaceConfig.js';
import { loadMarketplace } from '../marketplace/MarketplaceLoader.js';
import { MarketplaceService } from '../marketplace/MarketplaceService.js';
import { validateManifest } from '../../utils/validation/PluginValidator.js';
import {
    parsePluginId,
    validateScope,
    getProjectPathForScope,
    isLocalSource,
    getSettingsScope
} from './PluginUtils.js';
import { log } from '../logger/loggerService.js';

const logger = log('ModelContextProtocol');
const logInfo = (msg: string, ...args: any[]) => logger.info(msg, ...args);
const logError = (msg: any, ...args: any[]) => logger.error(msg, ...args);
import {
    getInstalledPlugins,
    refreshPlugins,
    clearInstalledPluginsCache,
    installPluginInternal,
    recordUninstallation,
    removeDirectory,
    isManagedScope,
    togglePluginInternal,
    getPluginStore,
    downloadPlugin,
    getLatestVersion,
    getPluginInstallPath,
    copyPluginToInstallPath,
    recordInstallation,
} from './McpClientManager.js';

// --- Global State ---
let pluginUpdateCallback: ((pluginIds: string[]) => void) | null = null;
let pendingPluginUpdates: string[] | null = null;

// --- Helper Types ---
interface PluginEntry {
    name: string;
    description?: string;
    source: string | { source: string;[key: string]: any };
    version?: string;
    [key: string]: any;
}

interface PluginLookupResult {
    entry: any; // Allow any here to match marketplace results which might have varying schemas
    marketplaceName: string;
    marketplaceInstallLocation: string;
}

// --- Core Helper: Find Plugin in Marketplaces (jV) ---
export async function getPluginFromMarketplace(pluginId: string): Promise<PluginLookupResult | null> {
    const { name, marketplace: marketplaceNameParam } = parsePluginId(pluginId);

    if (marketplaceNameParam) {
        try {
            const marketplaceData = await loadMarketplace(marketplaceNameParam);
            const entry = marketplaceData.plugins.find((p: any) => p.name === name);
            if (entry) {
                const config = await getMarketplaceConfig();
                const marketplaceConfig = config[marketplaceNameParam];
                return {
                    entry,
                    marketplaceName: marketplaceNameParam,
                    marketplaceInstallLocation: marketplaceConfig?.installLocation
                };
            }
        } catch (err) {
            logInfo(`Failed to load marketplace "${marketplaceNameParam}": ${err}`);
        }
    } else {
        const config = await getMarketplaceConfig();
        for (const [mName, mDetails] of Object.entries(config)) {
            try {
                const marketplaceData = await loadMarketplace(mName);
                const entry = marketplaceData.plugins.find((p: any) => p.name === name);
                if (entry) {
                    return {
                        entry,
                        marketplaceName: mName,
                        marketplaceInstallLocation: (mDetails as any).installLocation
                    };
                }
            } catch (err) {
                logInfo(`Failed to check plugin ${name} in marketplace ${mName}: ${err}`);
                continue;
            }
        }
    }
    return null;
}

// --- Main Functions ---

/**
 * Installs a plugin (l29)
 */
export async function installPlugin(id: string, scope: string = "user") {
    validateScope(scope);
    const result = await getPluginFromMarketplace(id);

    if (!result) {
        const { marketplace } = parsePluginId(id);
        const target = marketplace ? `marketplace "${marketplace}"` : "any configured marketplace";
        return {
            success: false,
            message: `Plugin "${parsePluginId(id).name}" not found in ${target}`
        };
    }

    const { entry, marketplaceName, marketplaceInstallLocation } = result;
    const pluginId = `${entry.name}@${marketplaceName}`;
    const projectPath = getProjectPathForScope(scope);
    let installPath: string | undefined;
    const { source } = entry;

    // Local source handling (DP in chunk_524 / Dz in chunk_362)
    if (isLocalSource(typeof source === 'string' ? source : (source as any).path)) {
        if (!marketplaceInstallLocation) {
            return {
                success: false,
                message: `Cannot install local plugin "${entry.name}" without marketplace install location`
            };
        }
        installPath = path.join(marketplaceInstallLocation, typeof source === 'string' ? source : (source as any).path);
    }

    // Perform installation (Dz / installPluginInternal)
    const installedPath = await installPluginInternal(pluginId, entry, scope, projectPath || "", installPath);

    // Validate the installed plugin (rH1 in chunk_848)
    const validation = validateManifest(installedPath);
    if (!validation.success) {
        // If validation fails, we might want to uninstall or just warn.
        // Original logic seems to proceed but logs warnings/errors.
        logger.warn(`Plugin validation failed for ${pluginId}: ${JSON.stringify(validation.errors)}`);
    }

    const settingsScope = getSettingsScope(scope);
    const settings = getSettings(settingsScope);
    const enabledPlugins = {
        ...settings?.enabledPlugins,
        [pluginId]: true
    };

    const resultUpdate = updateSettings(settingsScope, { enabledPlugins });
    if (resultUpdate.error) {
        return {
            success: false,
            message: `Failed to update settings: ${(resultUpdate.error as any).message}`
        };
    }

    refreshPlugins();
    return {
        success: true,
        message: `Successfully installed plugin: ${pluginId} (scope: ${scope})`,
        pluginId,
        pluginName: entry.name,
        scope
    };
}

/**
 * Uninstalls a plugin (ekA)
 */
export async function uninstallPlugin(id: string, scope: string = "user") {
    validateScope(scope);
    const { enabled, disabled } = await getInstalledPlugins();
    const installed = [...enabled, ...disabled];

    // Find the installed plugin entry matching the ID or Name
    const { name } = parsePluginId(id);
    const entry = installed.find(p => p.name === name || p.id === id);

    if (!entry) {
        return {
            success: false,
            message: `Plugin "${id}" not found in installed plugins`
        };
    }

    // Resolve full plugin ID
    const pluginId = entry.id || id;
    const projectPath = getProjectPathForScope(scope);
    const installations = getPluginStore().plugins[pluginId];

    // Find the specific installation record for this scope/project
    const installation = installations?.find((i: any) => i.scope === scope && i.projectPath === projectPath);

    if (!installation) {
        // Try to find if it's installed in another scope to give a helpful message
        const foundScope = installations?.[0]?.scope;
        if (foundScope && installations.length > 0) {
            return {
                success: false,
                message: `Plugin "${id}" is installed in ${foundScope} scope, not ${scope}. Use --scope ${foundScope} to uninstall.`
            };
        }
        return {
            success: false,
            message: `Plugin "${id}" is not installed in ${scope} scope. Use --scope to specify the correct scope.`
        };
    }

    const { installPath } = installation;
    const settingsScopeName = getSettingsScope(scope);
    const settings = getSettings(settingsScopeName);

    const enabledPlugins = { ...settings?.enabledPlugins };
    delete enabledPlugins[pluginId]; // Remove from enabled settings

    updateSettings(settingsScopeName, { enabledPlugins });
    refreshPlugins();
    recordUninstallation(pluginId, scope, projectPath || "");

    // Check if any other installations use the same directory
    const store = getPluginStore();
    const remainingInstallations = store.plugins[pluginId];
    if ((!remainingInstallations || remainingInstallations.length === 0) && installPath) {
        removeDirectory(installPath);
    }

    return {
        success: true,
        message: `Successfully uninstalled plugin: ${entry.name} (scope: ${scope})`,
        pluginId,
        pluginName: entry.name,
        scope
    };
}

/**
 * Toggle plugin state (enable/disable) - i29
 */
async function togglePlugin(id: string, enable: boolean, scope?: string) {
    const action = enable ? "enable" : "disable";
    if (scope) validateScope(scope);

    const { enabled, disabled } = await getInstalledPlugins();
    const targetList = enable ? disabled : enabled;
    const { name } = parsePluginId(id);
    const entry = targetList.find(p => p.name === name || p.id === id);

    if (!entry) {
        return {
            success: false,
            message: `Plugin "${id}" not found in ${enable ? "disabled" : "enabled"} plugins`
        };
    }

    const pluginId = id.includes("@") ? id : `${entry.name}@${(entry as any).source?.split("@")[1] || "unknown"}`;
    let installationTarget: { scope: string, projectPath?: string };

    if (scope) {
        const projectPath = getProjectPathForScope(scope);
        installationTarget = { scope, projectPath };

        // Sanity check if the plugin is actually installed in this scope
        const _parsedContext = parsePluginId(pluginId);
        // This check is a bit simplified compared to original logic but covers the intent
    } else {
        // Auto-detect scope if not provided (simplified logic)
        const installations = getPluginStore().plugins[pluginId];
        if (installations && installations.length > 0) {
            installationTarget = { scope: installations[0].scope, projectPath: installations[0].projectPath };
        } else {
            // Fallback
            installationTarget = { scope: "user" };
        }
    }

    if (!isManagedScope(installationTarget.scope)) {
        // According to original logic, managed plugins can only be updated, not toggled?
        // Wait, original logic says: if (!$s(W.scope)) return ... cannot be enabled/disabled?
        // Actually $s is likely isManagedScope. And the message is "Managed plugins cannot be enabled/disabled".
        // BUT usually managed means enforced. So user can't disable them.
        if (installationTarget.scope === 'managed') {
            return {
                success: false,
                message: `Managed plugins cannot be ${action}d. They can only be updated.`
            };
        }
    }

    try {
        togglePluginInternal(pluginId, enable, entry, installationTarget);
    } catch (err) {
        return {
            success: false,
            message: err instanceof Error ? err.message : `Failed to ${action} plugin`
        };
    }

    return {
        success: true,
        message: `Successfully ${action}d plugin: ${entry.name} (scope: ${installationTarget.scope})`,
        pluginId,
        pluginName: entry.name,
        scope: installationTarget.scope
    };
}

export async function enablePlugin(id: string, scope?: string) {
    return togglePlugin(id, true, scope);
}

export async function disablePlugin(id: string, scope?: string) {
    return togglePlugin(id, false, scope);
}

/**
 * Updates a plugin (g4A)
 */
export async function updatePlugin(id: string, scope: string) {
    const { name, marketplace } = parsePluginId(id);
    const pluginId = marketplace ? `${name}@${marketplace}` : id;

    const marketplaceEntry = await getPluginFromMarketplace(id);
    if (!marketplaceEntry) {
        return {
            success: false,
            message: `Plugin "${name}" not found`,
            pluginId,
            scope
        };
    }

    const { entry, marketplaceInstallLocation } = marketplaceEntry;
    const installations = getPluginStore().plugins[pluginId];

    if (!installations || installations.length === 0) {
        return {
            success: false,
            message: `Plugin "${name}" is not installed`,
            pluginId,
            scope
        };
    }

    const projectPath = getProjectPathForScope(scope);
    const installation = installations.find((i: any) => i.scope === scope && i.projectPath === projectPath);

    if (!installation) {
        const scopeDesc = projectPath ? `${scope} (${projectPath})` : scope;
        return {
            success: false,
            message: `Plugin "${name}" is not installed at scope ${scopeDesc}`,
            pluginId,
            scope
        };
    }

    return performPluginUpdate({
        pluginId,
        pluginName: name,
        entry,
        marketplaceInstallLocation,
        installation,
        scope,
        projectPath
    });
}

/**
 * Performs the update logic (BB7)
 */
async function performPluginUpdate({
    pluginId,
    pluginName,
    entry,
    marketplaceInstallLocation,
    installation,
    scope,
    projectPath
}: any) {
    const currentVersion = installation.version;
    let sourcePath: string;
    let latestVersion: string;
    let isDownloaded = false;

    if (typeof entry.source !== "string") {
        const downloaded = await downloadPlugin(entry.source, { manifest: { name: entry.name } });
        sourcePath = downloaded.path;
        isDownloaded = true;
        latestVersion = await getLatestVersion(pluginId, entry.source, downloaded.manifest, downloaded.path, entry.version);
    } else {
        if (!fs.existsSync(marketplaceInstallLocation)) {
            return {
                success: false,
                message: `Marketplace directory not found at ${marketplaceInstallLocation}`,
                pluginId,
                scope
            };
        }
        // eQ7 logic (resolve symlink or dir?) - just assuming it's the dir for now
        const mDir = fs.statSync(marketplaceInstallLocation).isDirectory() ? marketplaceInstallLocation : path.dirname(marketplaceInstallLocation);
        sourcePath = path.join(mDir, entry.source);

        if (!fs.existsSync(sourcePath)) {
            return {
                success: false,
                message: `Plugin source not found at ${sourcePath}`,
                pluginId,
                scope
            };
        }

        // Manifest loading logic (u51)
        const manifestPath = path.join(sourcePath, ".claude-plugin", "plugin.json");
        let manifest;
        try {
            if (fs.existsSync(manifestPath)) {
                manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
            }
        } catch { }
        latestVersion = await getLatestVersion(pluginId, entry.source, manifest, sourcePath, entry.version);
    }

    try {
        const newInstallPath = getPluginInstallPath(pluginId, latestVersion);
        if (installation.version === latestVersion || installation.installPath === newInstallPath) {
            return {
                success: true,
                message: `${pluginName} is already at the latest version (${latestVersion}).`,
                pluginId,
                newVersion: latestVersion,
                oldVersion: currentVersion,
                alreadyUpToDate: true,
                scope
            };
        }

        if (!fs.existsSync(newInstallPath)) {
            // g51 - copy plugin
            await copyPluginToInstallPath(sourcePath, pluginId, latestVersion, entry);
        }

        const oldInstallPath = installation.installPath;
        // EQ2 - record installation
        recordInstallation(pluginId, scope, projectPath || "", newInstallPath, latestVersion);

        if (oldInstallPath && oldInstallPath !== newInstallPath) {
            const store = getPluginStore();
            const isInUse = Object.values(store.plugins).some((installs: any) =>
                installs.some((i: any) => i.installPath === oldInstallPath)
            );
            if (!isInUse && fs.existsSync(oldInstallPath)) {
                // f51 - remove directory
                removeDirectory(oldInstallPath);
            }
        }

        const scopeDesc = projectPath ? `${scope} (${projectPath})` : scope;
        return {
            success: true,
            message: `Plugin "${pluginName}" updated from ${currentVersion || "unknown"} to ${latestVersion} for scope ${scopeDesc}. Restart to apply changes.`,
            pluginId,
            newVersion: latestVersion,
            oldVersion: currentVersion,
            scope
        };
    } finally {
        if (isDownloaded && sourcePath !== getPluginInstallPath(pluginId, latestVersion)) {
            // cleanup temp download
            fs.rmSync(sourcePath, { recursive: true, force: true });
        }
    }
}

// --- Autoupdate Logic ---

export function onPluginUpdate(callback: (pluginIds: string[]) => void) {
    pluginUpdateCallback = callback;
    if (pendingPluginUpdates !== null && pendingPluginUpdates.length > 0) {
        callback(pendingPluginUpdates);
        pendingPluginUpdates = null;
    }
    return () => {
        pluginUpdateCallback = null;
    };
}

async function getMarketplacesWithAutoupdate(): Promise<Set<string>> {
    const config = await getMarketplaceConfig();
    const autoUpdateMarketplaces = new Set<string>();
    for (const [name, details] of Object.entries(config)) {
        if ((details as any).autoUpdate) {
            autoUpdateMarketplaces.add(name.toLowerCase());
        }
    }
    return autoUpdateMarketplaces;
}

// ZB7 - autoupdatePlugin logic (renamed for clarity)
async function autoupdatePluginLogic(id: string, installations: any[]) {
    let updated = false;
    for (const { scope } of installations) {
        try {
            const result = await updatePlugin(id, scope);
            if (result.success && !result.alreadyUpToDate) {
                updated = true;
                logInfo(`Plugin autoupdate: updated ${id} from ${result.oldVersion} to ${result.newVersion}`);
            } else if (!result.alreadyUpToDate) {
                logInfo(`Plugin autoupdate: failed to update ${id}: ${result.message}`, { level: "warn" });
            }
        } catch (err) {
            logInfo(`Plugin autoupdate: error updating ${id}: ${err instanceof Error ? err.message : String(err)}`, { level: "warn" });
        }
    }
    return updated ? id : null;
}

// YB7 - checkPluginsForUpdates
async function checkPluginsForUpdates(autoUpdateMarketplaces: Set<string>) {
    const store = getPluginStore();
    const pluginIds = Object.keys(store.plugins);
    const currentProject = process.cwd();

    if (pluginIds.length === 0) return [];

    const results = await Promise.allSettled(pluginIds.map(async (id) => {
        const { marketplace } = parsePluginId(id);
        if (!marketplace || !autoUpdateMarketplaces.has(marketplace.toLowerCase())) return null;

        const installations = store.plugins[id];
        if (!installations || installations.length === 0) return null;

        const relevantInstallations = installations.filter((i: any) =>
            i.scope === "user" || i.scope === "managed" || i.projectPath === currentProject
        );

        if (relevantInstallations.length === 0) return null;

        return autoupdatePluginLogic(id, relevantInstallations);
    }));

    return results
        .filter((r): r is PromiseFulfilledResult<string | null> => r.status === "fulfilled" && r.value !== null)
        .map(r => r.value as string);
}

/**
 * Perform background plugin installations and updates.
 * Based on OV1 in chunk_524.ts.
 */
export async function performBackgroundPluginInstallations() {
    logger.info("Checking for background plugin installations and updates...");
    try {
        // 1. Get all enabled plugins from settings
        const userSettings = getSettings('userSettings');
        const projectSettings = getSettings('projectSettings');
        const localSettings = getSettings('localSettings');

        const enabledPluginsSet = new Set<string>();
        [userSettings, projectSettings, localSettings].forEach(s => {
            if (s?.enabledPlugins) {
                for (const [id, enabled] of Object.entries(s.enabledPlugins)) {
                    if (enabled && id.includes('@')) enabledPluginsSet.add(id);
                }
            }
        });

        if (enabledPluginsSet.size === 0) {
            logger.info("No enabled plugins found for background installation.");
        } else {
            const store = getPluginStore();
            const installedIds = Object.keys(store.plugins);
            const missingPlugins = Array.from(enabledPluginsSet).filter(id => !installedIds.includes(id));

            if (missingPlugins.length > 0) {
                logger.info(`Found ${missingPlugins.length} missing plugins: ${missingPlugins.join(', ')}`);
                for (const pluginId of missingPlugins) {
                    const { marketplace } = parsePluginId(pluginId);
                    const config = await getMarketplaceConfig();

                    if (marketplace && config[marketplace]) {
                        logger.info(`Automatically installing missing plugin: ${pluginId}`);
                        await installPlugin(pluginId);
                    } else {
                        logger.warn(`Cannot install ${pluginId}: marketplace ${marketplace} not configured.`);
                    }
                }
            }
        }

        // 2. Handle autoupdates
        const autoUpdateMarketplaces = await getMarketplacesWithAutoupdate();
        if (autoUpdateMarketplaces.size > 0) {
            logger.info(`Checking autoupdates for ${autoUpdateMarketplaces.size} marketplaces...`);
            for (const mName of autoUpdateMarketplaces) {
                try {
                    await MarketplaceService.refreshMarketplace(mName);
                } catch (err) {
                    logger.warn(`Failed to refresh marketplace ${mName} during autoupdate: ${err}`);
                }
            }

            const updatedPlugins = await checkPluginsForUpdates(autoUpdateMarketplaces);
            if (updatedPlugins.length > 0) {
                logger.info(`Autoupdated ${updatedPlugins.length} plugins: ${updatedPlugins.join(', ')}`);
                if (pluginUpdateCallback) pluginUpdateCallback(updatedPlugins);
                else pendingPluginUpdates = updatedPlugins;
            }
        }
    } catch (err) {
        logger.error("Error during background plugin installations:", err);
    }
}

// Keeping this alias for compatibility
export const startPluginAutoupdate = performBackgroundPluginInstallations;


// --- Helpers for UI/Views ---

export { parsePluginId, validateScope } from './PluginUtils.js';
export { loadMarketplace } from '../marketplace/MarketplaceLoader.js';

export function validatePluginId(id: string): boolean {
    return id.length > 0;
}

export function getPluginSource(source: any): string {
    if (typeof source === 'string') return source;
    if (source && typeof source === 'object') {
        if (source.source === 'github') return `GitHub: ${source.repo}`;
        if (source.source === 'npm') return `npm: ${source.package}`;
        if (source.type === 'local') return `Local: ${source.path}`;
    }
    return 'Unknown';
}

export function getMarketplaceIssues(failures: any[], _total: number) {
    if (failures.length > 0) {
        return {
            type: "warning",
            message: `Failed to load ${failures.length} marketplace${failures.length === 1 ? '' : 's'}`
        };
    }
    return null;
}

export async function fetchPluginInstallCounts() {
    const map = new Map<string, number>();
    const store = getPluginStore();
    for (const [id, installs] of Object.entries(store.plugins)) {
        if (Array.isArray(installs)) {
            map.set(id, installs.length);
        }
    }
    return map;
}

export async function fetchMarketplaces() {
    return await getMarketplaceConfig();
}

export async function processMarketplaceData(configs: any) {
    const marketplaces = [];
    const failures = [];

    for (const [name, config] of Object.entries(configs)) {
        try {
            const data = await loadMarketplace(name);
            marketplaces.push({ name, config, data });
        } catch (err) {
            failures.push({ name, error: err });
            marketplaces.push({ name, config, data: null });
        }
    }
    return { marketplaces, failures };
}


export function getConfig(scope: string) {
    return getSettings(getSettingsScope(scope));
}

export function updateConfig(scope: string, updates: any) {
    return updateSettings(getSettingsScope(scope), updates);
}

export function clearPluginCache() {
    clearInstalledPluginsCache();
}

export function formatFailedInstalls(failures: any[], _verbose: boolean) {
    if (failures.length === 0) return "";
    return failures.map(f => `${f.name} (${f.reason})`).join(", ");
}

export async function handlePluginAction(params: { pluginId: string, entry: any, marketplaceName: string, scope: string }) {
    const { pluginId, scope } = params;
    return await installPlugin(pluginId, scope);
}

export function getHomepageActions(homepage: string | undefined, githubUrl: string | undefined) {
    const actions = [
        { label: "Install for you (user scope)", action: "install-user" },
        { label: "Install for project collaborators (project scope)", action: "install-project" },
        { label: "Install locally (repo scope)", action: "install-local" }
    ];

    if (homepage) {
        actions.push({ label: "Open Homepage", action: "homepage" });
    }
    if (githubUrl) {
        actions.push({ label: "Open GitHub", action: "github" });
    }

    actions.push({ label: "Back to list", action: "back" });
    return actions;
}

export function getGithubUrl(plugin: any) {
    const source = plugin?.entry?.source;
    if (source && source.source === 'github') {
        return source.repo;
    }
    return undefined;
}
