import { finalizePluginInstallation } from "./PluginInstaller.js";
import { cachePluginToTemp } from "./PluginFetcher.js";
import { cacheAndLoadPlugin } from "./PluginLoader.js";
import { getSettings, updateSettings } from "../terminal/settings.js";
import { logEvent } from "../telemetry/TelemetryService.js";
import { InstalledPluginStore } from "./InstalledPluginStore.js";
import { registerLocalPlugin } from "./PluginInstaller.js";
import * as path from "path";
import * as fs from "fs";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

// Logic from chunk_363.ts (h51, t65)

function getEnabledPluginsFromSettings(destination: "user" | "project" = "user") {
    // Simplified access, ideally use getSettings(destination)
    return getSettings(destination === "user" ? "userSettings" : "projectSettings")?.enabledPlugins || {};
}

function updateEnabledPlugins(destination: "user" | "project", plugins: any) {
    updateSettings(destination === "user" ? "userSettings" : "projectSettings", {
        enabledPlugins: plugins
    });
}

export async function installPlugin(pluginId: string, entry: any, marketplaceName?: string, scope: "user" | "project" = "user") {
    try {
        let destination = scope;

        // Fetch and install
        // If entry.source is string (url/path) or object

        // If it's a marketplace install, we might have a specific location?
        // Logic from h51 follows finalizePluginInstallation

        await finalizePluginInstallation(pluginId, entry.manifest, scope, undefined, entry);

        const enabled = getEnabledPluginsFromSettings(scope);
        const newEnabled = {
            ...enabled,
            [pluginId]: true
        };

        updateEnabledPlugins(scope, newEnabled);

        logEvent("tengu_plugin_installed", {
            plugin_id: pluginId,
            marketplace_name: marketplaceName
        });

        // Trigger reload? sZ() calls b65 calls Ho, y51, ... (reload hooks, commands, etc)
        // For now return success

        return { success: true, message: `✓ Installed ${pluginId}. Restart Claude Code to load new plugins.` };
    } catch (error: any) {
        logEvent("tengu_plugin_install_failed", {
            plugin_id: pluginId,
            error: error.message
        });
        return { success: false, error: `Failed to install: ${error.message}` };
    }
}

function resolveInstalledLocation(pluginId: string): { entry: any, location: string } | null {
    // Check InstalledPluginStore
    const data = InstalledPluginStore.getAllInstalledPlugins();
    const entries = data.plugins[pluginId];
    if (entries && entries.length > 0) {
        // Sort by version or last updated?
        const latest = entries[entries.length - 1]; // Simplified
        if (latest && latest.installPath && fs.existsSync(latest.installPath)) {
            return { entry: latest, location: latest.installPath };
        }
    }
    return null;
}

export async function getEnabledPlugins(): Promise<{ enabled: any[], errors: any[] }> {
    const userEnabled = getEnabledPluginsFromSettings("user");
    const projectEnabled = getEnabledPluginsFromSettings("project");

    // Merge enabled (project overrides user? or union?)
    const allEnabled = { ...userEnabled, ...projectEnabled };
    const enabledList: any[] = [];
    const errors: any[] = [];

    for (const [pluginId, isEnabled] of Object.entries(allEnabled)) {
        if (!isEnabled) continue;

        try {
            const resolved = resolveInstalledLocation(pluginId);
            // If not resolved, we pass null location to let cacheAndLoadPlugin try to restore if it can (requires entry)
            // But we lack entry here if it's not in InstalledPluginStore?
            // Actually getEnabledPlugins should look up Marketplace (?) if missing locally?
            // chunk_363.ts t65 uses t3 (marketplace) to find entry.
            // Simplified: only load if installed for now.

            if (!resolved) {
                errors.push({
                    type: "plugin-not-found",
                    source: pluginId,
                    pluginId: pluginId
                });
                continue;
            }

            const result = await cacheAndLoadPlugin(resolved.entry, resolved.location, pluginId, true, errors);
            if (result && result.plugin) {
                enabledList.push(result.plugin);
            }
            if (result && result.errors) errors.push(...result.errors);

        } catch (e: any) {
            errors.push({
                type: "generic-error",
                source: pluginId,
                error: e.message
            });
        }
    }

    return { enabled: enabledList, errors };
}
