
import * as fs from "fs";
import * as path from "path";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";
import { InstalledPluginStore } from "./InstalledPluginStore.js";

// Logic from chunk_362.ts (e62)

const ORPHANED_AT_FILE = ".orphaned_at";
const ORPHAN_GRACE_PERIOD_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

function getCacheDir(): string {
    return path.join(getConfigDir(), "plugins", "cache");
}

function getOrphanedMarkerPath(dir: string): string {
    return path.join(dir, ORPHANED_AT_FILE);
}

function markOrphaned(dir: string) {
    try {
        const markerPath = getOrphanedMarkerPath(dir);
        if (!fs.existsSync(markerPath)) {
            fs.writeFileSync(markerPath, Date.now().toString(), "utf-8");
        }
    } catch (e) {
        console.error(`Failed to mark orphaned plugin at ${dir}:`, e);
    }
}

function clearOrphanedMark(dir: string) {
    try {
        const markerPath = getOrphanedMarkerPath(dir);
        if (fs.existsSync(markerPath)) {
            fs.unlinkSync(markerPath);
        }
    } catch (e) {
        console.error(`Failed to clear orphaned mark at ${dir}:`, e);
    }
}

function checkAndRemoveOrphan(dir: string, now: number) {
    const markerPath = getOrphanedMarkerPath(dir);
    if (!fs.existsSync(markerPath)) {
        markOrphaned(dir);
        return;
    }

    try {
        const content = fs.readFileSync(markerPath, "utf-8");
        const orphanedAt = parseInt(content.trim(), 10);
        if (!isNaN(orphanedAt) && (now - orphanedAt > ORPHAN_GRACE_PERIOD_MS)) {
            // Expired, delete
            console.log(`Removing orphaned plugin version at ${dir}`);
            fs.rmSync(dir, { recursive: true, force: true });
        }
    } catch (e) {
        console.error(`Failed to process orphaned plugin at ${dir}:`, e);
    }
}

function getActivePluginPaths(): Set<string> {
    const activePaths = new Set<string>();
    const data = InstalledPluginStore.getAllInstalledPlugins();

    for (const entries of Object.values(data.plugins)) {
        if (Array.isArray(entries)) {
            for (const entry of entries) {
                if (entry.installPath) {
                    activePaths.add(path.resolve(entry.installPath));
                }
            }
        }
    }
    return activePaths;
}

function getSubdirectories(dir: string): string[] {
    try {
        return fs.readdirSync(dir, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => path.join(dir, d.name));
    } catch {
        return [];
    }
}

function removeIfEmpty(dir: string) {
    try {
        if (fs.readdirSync(dir).length === 0) {
            fs.rmdirSync(dir);
        }
    } catch { }
}

export async function cleanupOrphanedPlugins() {
    try {
        const activePaths = getActivePluginPaths();
        const cacheDir = getCacheDir();
        if (!fs.existsSync(cacheDir)) return;

        const now = Date.now();

        // Structure is cacheDir / pluginName / version
        // Or cacheDir / flattened_name / version (based on my PluginFetcher logic)

        const pluginDirs = getSubdirectories(cacheDir);
        for (const pluginDir of pluginDirs) {
            let pluginDirHasVersions = false;
            const versionDirs = getSubdirectories(pluginDir);

            for (const versionDir of versionDirs) {
                const resolved = path.resolve(versionDir);
                if (activePaths.has(resolved)) {
                    clearOrphanedMark(versionDir);
                    pluginDirHasVersions = true;
                } else {
                    checkAndRemoveOrphan(versionDir, now);
                    // Check if it still exists
                    if (fs.existsSync(versionDir)) {
                        pluginDirHasVersions = true;
                    }
                }
            }

            if (!pluginDirHasVersions) {
                removeIfEmpty(pluginDir);
            }
        }

    } catch (e) {
        console.error("Plugin cleanup failed:", e);
    }
}
