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
        if (!fs.existsSync(filePath)) {
            cachedPlugins = { version: 2, plugins: {} };
            return cachedPlugins;
        }

        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const data = JSON.parse(content);

            // Basic migration check (Stubbed)
            if (data.version === 1) {
                console.log("[Plugins] Migrating plugin database from V1 to V2...");
                data.version = 2;
                // actual migration logic would go here
            }

            cachedPlugins = data;
            return cachedPlugins!;
        } catch (error) {
            console.error("[Plugins] Failed to load plugin database:", error);
            cachedPlugins = { version: 2, plugins: {} };
            return cachedPlugins;
        }
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
