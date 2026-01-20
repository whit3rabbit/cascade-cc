
import * as fs from "fs";
import * as path from "path";
import { InstalledPluginsV1Schema } from "./MarketplaceSchemas.js";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";
import { getPluginInstallPath } from "../mcp/McpClientManager.js";
import { log } from "../logger/loggerService.js";

const logger = log("MarketplaceService");

// da()
function getAllPluginsInstallDir() {
    return path.join(getConfigDir(), "plugins", "installed");
}

function getPluginsDir(): string {
    return path.join(getConfigDir(), "plugins");
}

function getInstalledPluginsPath() {
    return path.join(getPluginsDir(), "installed_plugins.json");
}

function getInstalledPluginsV2Path() {
    return path.join(getPluginsDir(), "installed_plugins_v2.json");
}

let migrationRun = false;

// kA5
export function migrateInstalledPluginsDb() {
    if (migrationRun) return;

    const installedV1Path = getInstalledPluginsPath();
    const installedV2Path = getInstalledPluginsV2Path();

    try {
        const hasV2 = fs.existsSync(installedV2Path);
        const hasV1 = fs.existsSync(installedV1Path);

        if (hasV2) {
            fs.renameSync(installedV2Path, installedV1Path);
            logger.info("Renamed installed_plugins_v2.json to installed_plugins.json");
            const db = JSON.parse(fs.readFileSync(installedV1Path, "utf-8"));
            cleanupLegacyCache(db);
        } else if (hasV1) {
            const content = fs.readFileSync(installedV1Path, "utf-8");
            const json = JSON.parse(content);
            const version = typeof json?.version === "number" ? json.version : 1;

            if (version === 1) {
                const v1 = InstalledPluginsV1Schema.parse(json);
                const v2 = convertInstalledPluginsV1ToV2(v1);
                fs.writeFileSync(installedV1Path, JSON.stringify(v2, null, 2), { encoding: "utf-8", flush: true });
                logger.info(`Converted installed_plugins.json from V1 to V2 format (${Object.keys(v1.plugins).length} plugins)`);
                cleanupLegacyCache(v2);
            }
        }
        migrationRun = true;
    } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        logger.error(`Failed to migrate plugin files: ${msg}`);
        // Log telemetry or additional error info if needed
        migrationRun = true;
    }
}

// HQ2
function cleanupLegacyCache(db: any) {
    const installDir = getAllPluginsInstallDir();
    if (!fs.existsSync(installDir)) return;

    try {
        const activePaths = new Set<string>();
        // Iterate through all plugins and their installations to collect active paths
        for (const installations of Object.values(db.plugins)) {
            if (Array.isArray(installations)) {
                for (const install of installations) {
                    if (install.installPath) {
                        activePaths.add(install.installPath);
                    }
                }
            }
        }

        const entries = fs.readdirSync(installDir, { withFileTypes: true });
        for (const entry of entries) {
            if (!entry.isDirectory()) continue;

            const pluginDir = path.join(installDir, entry.name);

            // Check if this directory is in use (it might be a versioned dir, or a plugin container dir)
            // Logic from HQ2: it seems to check 2 levels deep?
            // "if Q.readdirSync(X).some(...) continue" - checking if subdirs exist?
            // "if !G.has(X)" - assumes X is the installPath.
            // If getPluginInstallPath returns ".../plugins/installed/pluginId/version", then X is "pluginId".
            // The activePaths would contain ".../pluginId/version".

            // Let's rely on strict path checking if we can, or simple logic:
            // The original code HQ2 logic:
            // Iterate Y of Z (entries in installDir).
            // X = join(B, J) (plugin dir).
            // Check if directory has subdirectories (versions).
            // if (Q.readdirSync(X).some((K) => ...)) continue;
            // The check seems to be: if it has any active subdirectories, keep it?

            // Simplified cleanup for now:
            // If the plugin dir is not referenced by any install path, we might want to verify deeper.
            // But strict matching is safer.
        }
    } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        logger.warn(`Failed to clean up legacy cache: ${msg}`);
    }
}

export function readInstalledPluginsFile() {
    const p = getInstalledPluginsPath();
    if (!fs.existsSync(p)) return null;
    const content = fs.readFileSync(p, "utf-8");
    const json = JSON.parse(content);
    return {
        version: typeof json?.version === "number" ? json.version : 1,
        data: json
    };
}

// A70
export function convertInstalledPluginsV1ToV2(v1: any): any {
    const v2Plugins: Record<string, any[]> = {};
    for (const [id, data] of Object.entries(v1.plugins)) {
        const d = data as any;
        // Fk(B, G.version) -> getPluginInstallPath
        const installPath = getPluginInstallPath(id, d.version);

        v2Plugins[id] = [{
            scope: "user",
            installPath: installPath,
            version: d.version,
            installedAt: d.installedAt,
            lastUpdated: d.lastUpdated,
            gitCommitSha: d.gitCommitSha,
            isLocal: d.isLocal
        }];
    }
    return { version: 2, plugins: v2Plugins };
}
