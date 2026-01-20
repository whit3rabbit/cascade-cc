
import * as fs from "fs";
import * as path from "path";
import { cachePluginToTemp } from "./PluginFetcher.js";
import { InstalledPluginStore } from "./InstalledPluginStore.js";
import { runGitCommand } from "../../utils/git/GitUtils.js";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

// Logic from chunk_362.ts (Dz, Z82)

async function getGitCommitSha(cwd: string): Promise<string> {
    try {
        const result = await runGitCommand(["rev-parse", "HEAD"], cwd);
        if (result.code === 0) return result.stdout.trim();
    } catch (e) {
        // ignore
    }
    return "unknown";
}

function getInstallPath(pluginName: string, version: string): string {
    // Standard installation path generally matches the cache path for managed plugins
    return path.join(getConfigDir(), "plugins", "cache", pluginName.replace(/[^a-zA-Z0-9\-_]/g, "-"), version.replace(/[^a-zA-Z0-9\-_.]/g, "-"));
}

export async function finalizePluginInstallation(pluginId: string, manifest: any, scope: string = "user", projectPath?: string, existingSource?: any): Promise<string> {
    // 1. Fetch/Cache the plugin
    // existingSource is likely the original entry (url, repo, etc)
    const result = await cachePluginToTemp(existingSource, { manifest });

    // 2. Get git info if available
    const gitSha = await getGitCommitSha(result.path);
    const installedAt = new Date().toISOString();
    const version = manifest.version || (existingSource && existingSource.ref) || "unknown";

    // 3. Update persistent store
    InstalledPluginStore.upsertInstalledPlugin(pluginId, {
        version,
        installedAt,
        lastUpdated: installedAt,
        installPath: result.path,
        gitCommitSha: gitSha,
        isLocal: typeof existingSource === "string"
    }, scope, projectPath);

    return result.path;
}

export function registerLocalPlugin(pluginId: string, version: string, installPath: string, scope: string = "user", projectPath?: string) {
    const installedAt = new Date().toISOString();
    InstalledPluginStore.upsertInstalledPlugin(pluginId, {
        version: version || "unknown",
        installedAt,
        lastUpdated: installedAt,
        installPath,
        isLocal: true
    }, scope, projectPath);
}
