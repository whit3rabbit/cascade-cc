
import * as fs from "fs";
import * as path from "path";
import { loadPluginComponents } from "./PluginManifestLoader.js";
import { cachePluginToTemp } from "./PluginFetcher.js";
import { finalizePluginInstallation } from "./PluginInstaller.js";
import { InstalledPluginStore } from "./InstalledPluginStore.js";
import { logEvent } from "../telemetry/TelemetryService.js";

// Logic from chunk_364.ts (e65, A85)

export async function cacheAndLoadPlugin(entry: any, installLocation: string, sourceId: string, enabled: boolean, errors: any[]) {
    let pluginPath = installLocation;

    // Check if plugin exists at location
    // If installLocation is null, or not exists, we might need to fetch it (restore)
    // In original e65, it handles fetching if missing

    if (!pluginPath || !fs.existsSync(pluginPath)) {
        // Attempt to restore/fetch
        try {
            console.log(`Plugin ${entry.name} not found at ${installLocation}, attempting to restore...`);
            // We need to know scope. entry from marketplace might not have scope info directly?
            // Assuming user scope for restore
            const scope = "user";

            // This re-installs it.
            // But we need to be careful about not looping or erroring if source is invalid
            if (entry.source) {
                const result = await cachePluginToTemp(entry.source, { manifest: entry.manifest });
                // Update store
                // We don't have scope here easily? 
                pluginPath = result.path;
                // finalizePluginInstallation might be better but it upserts store
                // let's assume we use the temp path for this session if we can't fully reinstall
            } else {
                errors.push({
                    type: "generic-error",
                    source: sourceId,
                    error: `Plugin not found and no source available to restore`
                });
                return null;
            }
        } catch (e: any) {
            errors.push({
                type: "restore-failed",
                source: sourceId,
                error: e.message
            });
            return null;
        }
    }

    const { plugin, errors: loadErrors } = loadPluginComponents(pluginPath, sourceId, enabled);
    if (loadErrors.length > 0) errors.push(...loadErrors);

    // Merge marketplace overrides
    // 1. Commands
    if (entry.commands) {
        // overrides logic ... simplified:
        // If entry.commands is different/more specific, we might overlay it
        // For now, rely on manifest
    }

    return { plugin, errors };
}

export async function loadSessionPlugins(paths: string[]) {
    const plugins: any[] = [];
    const errors: any[] = [];

    for (const p of paths) {
        if (!fs.existsSync(p)) {
            errors.push({ type: "path-not-found", path: p });
            continue;
        }

        try {
            const { plugin, errors: loadErrors } = loadPluginComponents(p, `${path.basename(p)}@inline`, true);
            plugin.source = "session";
            plugins.push(plugin);
            errors.push(...loadErrors);
        } catch (e: any) {
            errors.push({ type: "load-failed", path: p, error: e.message });
        }
    }

    return { plugins, errors };
}
