
// Logic from chunk_588.ts (Configuration & Settings Management)

import { readFileSync, writeFileSync, existsSync, statSync, mkdirSync, copyFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { randomUUID } from "node:crypto";

const CONFIG_FILE = join(process.env.HOME || "", ".claude", "settings.json");
const BACKUP_FILE = CONFIG_FILE + ".backup";

let cachedConfig: any = null;
let lastMtime = 0;

/**
 * Reads the configuration from disk with caching.
 */
export function getConfig() {
    try {
        const stats = existsSync(CONFIG_FILE) ? statSync(CONFIG_FILE) : null;
        if (cachedConfig && stats && stats.mtimeMs <= lastMtime) {
            return cachedConfig;
        }

        if (!stats) return { userID: randomUUID(), projects: {} };

        const content = readFileSync(CONFIG_FILE, "utf-8");
        cachedConfig = JSON.parse(content);
        lastMtime = stats.mtimeMs;
        return cachedConfig;
    } catch (err) {
        console.error("Failed to read config, returning defaults:", err);
        return { projects: {} };
    }
}

/**
 * Atomic write to configuration file.
 */
export function updateConfig(updater: (config: any) => any) {
    const config = getConfig();
    const newConfig = updater(config);

    if (JSON.stringify(config) === JSON.stringify(newConfig)) return;

    try {
        const configDir = dirname(CONFIG_FILE);
        if (!existsSync(configDir)) mkdirSync(configDir, { recursive: true });

        // Backup existing config
        if (existsSync(CONFIG_FILE)) copyFileSync(CONFIG_FILE, BACKUP_FILE);

        writeFileSync(CONFIG_FILE, JSON.stringify(newConfig, null, 2));
        cachedConfig = newConfig;
        lastMtime = Date.now();
    } catch (err) {
        console.error("Failed to save config:", err);
    }
}

/**
 * Returns the current project-specific configuration.
 */
export function getProjectConfig(projectId: string) {
    const config = getConfig();
    return config.projects?.[projectId] || { allowedTools: [] };
}

/**
 * Updates context for a specific project.
 */
export function updateProjectConfig(projectId: string, updater: (projConfig: any) => any) {
    updateConfig(config => {
        const projects = config.projects || {};
        const projConfig = projects[projectId] || { allowedTools: [] };
        projects[projectId] = updater(projConfig);
        return { ...config, projects };
    });
}

/**
 * Checks if auto-updates are disabled and why.
 */
export function getAutoupdateDisabledReason(): string | null {
    if (process.env.DISABLE_AUTOUPDATER) return "DISABLE_AUTOUPDATER set";
    if (process.env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC) return "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC set";

    const config = getConfig();
    if (config.autoUpdates === false) return "Disabled in settings";

    return null;
}
