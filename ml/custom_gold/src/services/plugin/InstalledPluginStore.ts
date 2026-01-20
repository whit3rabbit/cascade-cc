import * as fs from "fs";
import * as path from "path";
import { InstalledPluginsV2Schema, InstalledPluginsDbSchema, PluginInstallationV2Schema } from "../marketplace/MarketplaceSchemas.js";
import { convertInstalledPluginsV1ToV2 } from "../marketplace/InstalledPluginsMigration.js";

function getInstalledPluginsPath(): string {
    return path.join(process.cwd(), ".claude", "plugins", "installed_plugins.json"); // Placeholder
}

let cachedPlugins: any = null;

export class InstalledPluginStore {
    static getAllInstalledPlugins() {
        if (cachedPlugins) return cachedPlugins;
        const filePath = getInstalledPluginsPath();
        if (!fs.existsSync(filePath)) {
            cachedPlugins = { version: 2, plugins: {} };
            return cachedPlugins;
        }

        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const json = JSON.parse(content);
            if ((json.version === undefined ? 1 : json.version) === 1) {
                // Determine V1
                const v1 = InstalledPluginsDbSchema.parse(json);
                if ('plugins' in v1 && !Array.isArray(Object.values(v1.plugins)[0])) {
                    // It's V1
                    const v2 = convertInstalledPluginsV1ToV2(v1);
                    cachedPlugins = v2;
                } else {
                    cachedPlugins = InstalledPluginsV2Schema.parse(json);
                }
            } else {
                cachedPlugins = InstalledPluginsV2Schema.parse(json);
            }
            return cachedPlugins;
        } catch (error) {
            console.error("Failed to load installed plugins:", error);
            cachedPlugins = { version: 2, plugins: {} };
            return cachedPlugins;
        }
    }

    static saveInstalledPlugins(data: any) {
        const filePath = getInstalledPluginsPath();
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
        cachedPlugins = data;
    }

    static removeInstalledPlugin(pluginId: string, scope: string, projectPath?: string) {
        const data = this.getAllInstalledPlugins();
        const entries = data.plugins[pluginId];
        if (!entries) return;

        const newEntries = entries.filter((e: any) => !(e.scope === scope && e.projectPath === projectPath));
        if (newEntries.length === 0) {
            delete data.plugins[pluginId];
        } else {
            data.plugins[pluginId] = newEntries;
        }
        this.saveInstalledPlugins(data);
    }

    static upsertInstalledPlugin(pluginId: string, entry: any, scope: string, projectPath?: string) {
        const data = this.getAllInstalledPlugins();
        const existingEntries = data.plugins[pluginId] || [];

        const newEntry = {
            ...entry,
            scope,
            projectPath
        };

        const index = existingEntries.findIndex((e: any) => e.scope === scope && e.projectPath === projectPath);
        if (index >= 0) {
            existingEntries[index] = newEntry;
        } else {
            existingEntries.push(newEntry);
        }

        data.plugins[pluginId] = existingEntries;
        this.saveInstalledPlugins(data);
    }
}
