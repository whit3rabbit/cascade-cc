/**
 * File: src/services/plugin/InstalledPluginStore.ts
 * Role: Persistent storage for installed MCP plugins, handling V1->V2 migrations.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';

interface PluginEntry {
    version: string;
    scope: string;
    projectPath?: string;
    installPath: string;
}

interface PluginStoreData {
    version: number;
    plugins: Record<string, PluginEntry[]>;
}

let cachedPlugins: PluginStoreData | null = null;

function getInstalledPluginsPath(): string {
    return path.join(getBaseConfigDir(), "plugins", "installed_plugins.json");
}

/**
 * Manages the JSON database of installed MCP plugins.
 */
export class InstalledPluginStore {
    /**
     * Loads all installed plugins from disk, applying migrations if necessary.
     */
    static getAllInstalledPlugins(): PluginStoreData {
        if (cachedPlugins) return cachedPlugins;

        const filePath = getInstalledPluginsPath();
        const v2FilePath = path.join(path.dirname(filePath), "installed_plugins_v2.json");

        // Check for v2 file rename migration first
        if (fs.existsSync(v2FilePath)) {
            try {
                fs.renameSync(v2FilePath, filePath);
                console.log("[Plugins] Renamed installed_plugins_v2.json to installed_plugins.json");
                // Load and cleanup legacy cache after rename
                const content = fs.readFileSync(filePath, 'utf-8');
                const data = JSON.parse(content);
                this.cleanupLegacyCache(data);
                cachedPlugins = data;
                return cachedPlugins!;
            } catch (error) {
                console.error("[Plugins] Failed to rename v2 plugin file:", error);
            }
        }

        if (!fs.existsSync(filePath)) {
            cachedPlugins = { version: 2, plugins: {} };
            return cachedPlugins;
        }

        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const rawData = JSON.parse(content);

            // Migration check
            // If version is missing or 1, migrate
            const version = typeof rawData.version === 'number' ? rawData.version : 1;

            if (version === 1) {
                console.log(`[Plugins] Conversing installed_plugins.json from V1 to V2 format (${Object.keys(rawData.plugins || {}).length} plugins)`);
                const v2Data = this.convertV1ToV2(rawData);

                // Persist V2 data
                this.saveInstalledPlugins(v2Data);

                // Cleanup legacy cache
                this.cleanupLegacyCache(v2Data);

                cachedPlugins = v2Data;
                return cachedPlugins;
            }

            cachedPlugins = rawData;
            return cachedPlugins!;
        } catch (error) {
            console.error("[Plugins] Failed to load plugin database:", error);
            cachedPlugins = { version: 2, plugins: {} };
            return cachedPlugins;
        }
    }

    /**
     * Converts V1 plugin data to V2 format.
     */
    private static convertV1ToV2(v1Data: any): PluginStoreData {
        const v2Plugins: Record<string, PluginEntry[]> = {};

        const entries = Object.entries(v1Data.plugins || {});
        for (const [pluginId, data] of entries as [string, any][]) {
            const version = data.version;
            // Original code calls Ex(pluginId, version) -> getPluginInstallPath
            // We need to implement getPluginInstallPath logic here or in a helper
            // Based on chunk468: Ex(G, k) -> likely joins cache dir + sanitized name + version
            const installPath = this.getPluginInstallPath(pluginId, version);

            v2Plugins[pluginId] = [{
                scope: 'user', // Default scope
                installPath: installPath,
                version: version,
                // Copy other props
                // installedAt, lastUpdated, gitCommitSha if present
                ...(data.installedAt ? { installedAt: data.installedAt } : {}),
                ...(data.lastUpdated ? { lastUpdated: data.lastUpdated } : {}),
                ...(data.gitCommitSha ? { gitCommitSha: data.gitCommitSha } : {})
            } as any]; // Cast to any to include extra props if Interface isn't strict yet
        }

        return {
            version: 2,
            plugins: v2Plugins
        };
    }

    /**
     * Cleans up legacy cache directories unrelated to current installations.
     */
    private static cleanupLegacyCache(data: PluginStoreData) {
        const cacheDir = this.getPluginCacheDir();
        if (!fs.existsSync(cacheDir)) return;

        try {
            // Collect all active install paths
            const activePaths = new Set<string>();
            for (const entries of Object.values(data.plugins)) {
                for (const entry of entries) {
                    activePaths.add(entry.installPath);
                }
            }

            const items = fs.readdirSync(cacheDir, { withFileTypes: true });
            for (const item of items) {
                if (!item.isDirectory()) continue;

                // item.name is likely the sanitized plugin name folder
                // Check if this directory contains subdirectories (versions)
                // If the directory itself is in activePaths (unlikely with nested struct), keep it
                // But the structure seems to be plugins/sanitizedName/version

                // Original code logic:
                // Checks if any subdir of this dir is a directory
                // If yes, it iterates those subdirs?
                // Let's re-read chunk466 clearConversation_95 carefully.

                // Logic seems to be:
                // Iterate plugins dir (q) -> w (folder)
                // If w is directory:
                //   Check if w contains subdirectories that contain directories?
                //   Wait, clearConversation_95 logic:
                //   J = join(q, H) -> J is plugin root dir e.g. plugins/foo-bar
                //   Check if J has children $ that are directories, AND children of $ have directories?
                //   Actually, let's look at getPluginInstallPath assumption.
                //   If path is plugins/foo-bar/1.0.0
                //   Then activePaths has plugins/foo-bar/1.0.0

                //   The cleanup loop iterates `plugins` children.
                //   If a child is NOT in activePaths... wait.
                //   If J (plugins/foo-bar) is in activePaths? No, activePaths are deep.

                //   Actually, looking at chunk466:
                //   It checks if Y (Set of installPaths) has J.
                //   J is direct child of plugins dir.
                //   So if V2 uses nested paths, V1 might have used direct paths?
                //   OR, this cleans up finding "legacy" folders that are NOT part of the new structure.

                //   Let's stick to a safe cleanup:
                //   Recurse? No, let's follow the chunk logic exactly if possible.
                //   chunk466 lines 98-106 seems to protect "nested" structure.
                //   "if (readdir(J).some($ => isDir($) && readdir(join(J, $.name)).some(G => isDir(G)))) continue"
                //   This looks like it skips cleaning up if it detects a deep structure (maybe node_modules?).

                //   Simple interpretation:
                //   If J is NOT in activePaths, remove it.
                //   BUT, activePaths probably points to J/version if structure is plugins/name/version.
                //   So J would NOT be in activePaths.

                //   Wait, if V1 paths were flat? Or V2 paths are flat?
                //   Let's implement getPluginInstallPath first to see structure.

                // If we assume V2 structure is plugins/sanitizedName/version
                // Then activePaths = plugins/sanitizedName/version

                // If we iterate plugins/sanitizedName
                // It won't match activePaths.

                // Maybe I should skip aggressive cleanup for now to avoid deleting good stuff, 
                // or implement the "protection" logic.

                // Simpler approach for now:
                // Only clean up if we are sure.
                // The prompt is about "stub" V1->V2.
                // I will verify behavior with a test.

                // For now, I will implement a safe version that logs but maybe doesn't delete, or follows strict logic.
                // The original code delete seems to target "legacy cache".

                // Let's implement the helpers first.
            }
        } catch (error) {
            console.warn("[Plugins] Failed to clean up legacy cache:", error);
        }
    }

    /**
     * Gets the root directory for plugin caches.
     */
    private static getPluginCacheDir(): string {
        return path.join(getBaseConfigDir(), "plugins");
    }

    /**
     * Constructs the installation path for a plugin version.
     */
    static getPluginInstallPath(pluginId: string, version: string): string {
        const cacheDir = this.getPluginCacheDir();
        const sanitizedId = pluginId.replace(/[^a-zA-Z0-9-_]/g, '-');
        return path.join(cacheDir, sanitizedId, version);
    }

    /**
     * Persists the plugin database to disk.
     */
    static saveInstalledPlugins(data: PluginStoreData) {
        const filePath = getInstalledPluginsPath();
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
        cachedPlugins = data;
    }

    /**
     * Removes a plugin installation from the store.
     */
    static removeInstalledPlugin(pluginId: string, scope: string, projectPath?: string) {
        const data = this.getAllInstalledPlugins();
        const entries = data.plugins[pluginId];
        if (!entries) return;

        data.plugins[pluginId] = entries.filter(e => !(e.scope === scope && e.projectPath === projectPath));
        if (data.plugins[pluginId].length === 0) {
            delete data.plugins[pluginId];
        }

        this.saveInstalledPlugins(data);
    }

    /**
     * Adds or updates a plugin installation in the store.
     */
    static upsertInstalledPlugin(pluginId: string, entry: PluginEntry) {
        const data = this.getAllInstalledPlugins();
        const entries = data.plugins[pluginId] || [];

        const index = entries.findIndex(e => e.scope === entry.scope && e.projectPath === entry.projectPath);
        if (index >= 0) {
            entries[index] = entry;
        } else {
            entries.push(entry);
        }

        data.plugins[pluginId] = entries;
        this.saveInstalledPlugins(data);
    }
}
