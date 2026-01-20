import * as fs from "node:fs";
import * as path from "node:path";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";
import { log } from "../logger/loggerService.js";
import { settingsWatcher } from "./SettingsWatcher.js";
import { SettingsSchema } from "../settings/SettingsSchema.js";

const logger = log("settings");

export type SettingsSource =
    | "userSettings"
    | "projectSettings"
    | "localSettings"
    | "policySettings"
    | "flagSettings";

export const SETTING_SOURCES: SettingsSource[] = [
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
];

export interface Settings {
    permissions?: {
        defaultMode?: "prompt" | "allow" | "deny";
        allow?: string[];
        deny?: string[];
        ask?: string[];
        additionalDirectories?: string[];
        disableBypassPermissionsMode?: "disable";
    };
    [key: string]: any;
}

/**
 * Returns the path to the settings file for a given source.
 */
export function getSettingsPath(source: SettingsSource): string {
    const home = process.env.HOME || "";
    const configDir = getConfigDir();
    const projectDir = process.cwd();

    switch (source) {
        case "userSettings":
            return path.join(configDir, "settings.json");
        case "projectSettings":
            return path.join(projectDir, ".claude", "settings.json");
        case "localSettings":
            return path.join(projectDir, ".claude", "settings.local.json");
        case "policySettings":
            return "/etc/claude/managed-settings.json";
        case "flagSettings":
            // This is usually passed via CLI, not a file, but some parts of the code might expect a path
            // In chunk_590 it returns tF1() which might be a temp file or just null
            return "";
        default:
            return "";
    }
}

/**
 * Reads settings from a specific source.
 */
export function getSettings(source: SettingsSource): Settings {
    if (source === "policySettings") {
        // Logic for remote/managed policy would go here
    }

    const settingsPath = getSettingsPath(source);
    if (!settingsPath || !fs.existsSync(settingsPath)) return {};

    try {
        const content = fs.readFileSync(settingsPath, "utf-8");
        if (!content.trim()) return {};

        // In original code: R5(content, !1) which is JSON with comments
        const parsed = JSON.parse(content);

        const result = SettingsSchema.safeParse(parsed);
        if (!result.success) {
            logger.warn(`Validation failed for settings at ${settingsPath}: ${result.error.message}`);
            // Fallback to raw parsed if validation fails but syntax is ok? 
            // Original code returns Y.data if success, else errors.
            return parsed;
        }
        return result.data;
    } catch (err) {
        logger.error(`Failed to read settings from ${settingsPath}: ${err}`);
        return {};
    }
}

/**
 * Updates settings for a specific source.
 */
export function updateSettings(source: SettingsSource, updates: Partial<Settings>): { error: Error | null } {
    if (source === "policySettings" || source === "flagSettings") {
        return { error: new Error(`Cannot write to ${source}`) };
    }

    const settingsPath = getSettingsPath(source);
    const dir = path.dirname(settingsPath);

    try {
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        const current = getSettings(source);
        const next = deepMerge(current, updates);

        settingsWatcher.markInternalWrite(source);

        // Write atomically
        writeAtomic(settingsPath, JSON.stringify(next, null, 2) + "\n");

        return { error: null };
    } catch (err) {
        logger.error(`Failed to update settings at ${settingsPath}: ${err}`);
        return { error: err instanceof Error ? err : new Error(String(err)) };
    }
}

/**
 * Merges settings from all sources based on priority.
 */
export function mergeSettings(): Settings {
    const sources: SettingsSource[] = ["policySettings", "userSettings", "projectSettings", "localSettings"];
    let merged: Settings = {};

    for (const source of sources) {
        const sourceSettings = getSettings(source);
        merged = deepMerge(merged, sourceSettings);
    }

    return merged;
}

/**
 * Subscribes to settings changes.
 */
export function subscribeToSettings(callback: (source: SettingsSource) => void): () => void {
    return settingsWatcher.subscribe(callback);
}


/**
 * Returns a short description for a setting source.
 */
export function getSourceDescription(source: SettingsSource): string {
    switch (source) {
        case "userSettings":
            return "user";
        case "projectSettings":
            return "project";
        case "localSettings":
            return "project, gitignored";
        case "flagSettings":
            return "cli flag";
        case "policySettings":
            return "managed";
        default:
            return source;
    }
}

/**
 * Returns a capitalized label for a setting source.
 */
export function getSourceLabel(source: SettingsSource | "plugin" | "built-in"): string {
    switch (source) {
        case "userSettings":
            return "User";
        case "projectSettings":
            return "Project";
        case "localSettings":
            return "Local";
        case "flagSettings":
            return "Flag";
        case "policySettings":
            return "Managed";
        case "plugin":
            return "Plugin";
        case "built-in":
            return "Built-in";
        default:
            return source;
    }
}

/**
 * Returns a detailed description for a setting source.
 */
export function getSourceLongDescription(source: string): string {
    switch (source) {
        case "userSettings":
            return "user settings";
        case "projectSettings":
            return "shared project settings";
        case "localSettings":
            return "project local settings";
        case "flagSettings":
            return "command line arguments";
        case "policySettings":
            return "enterprise managed settings";
        case "cliArg":
            return "CLI argument";
        case "command":
            return "command configuration";
        case "session":
            return "current session";
        default:
            return source;
    }
}

/**
 * Returns a UI-friendly title for a setting source.
 */
export function getSourceDisplayTitle(source: string): string {
    switch (source) {
        case "userSettings":
            return "User settings";
        case "projectSettings":
            return "Shared project settings";
        case "localSettings":
            return "Project local settings";
        case "flagSettings":
            return "Command line arguments";
        case "policySettings":
            return "Enterprise managed settings";
        case "cliArg":
            return "CLI argument";
        case "command":
            return "Command configuration";
        case "session":
            return "Current session";
        default:
            return source;
    }
}

/**
 * Parses a comma-separated source string into an array of SettingsSource.
 */
export function parseSettingsSources(sourcesStr: string): SettingsSource[] {
    if (!sourcesStr) return [];
    const parts = sourcesStr.split(",").map(s => s.trim());
    const result: SettingsSource[] = [];
    for (const part of parts) {
        switch (part) {
            case "user":
                result.push("userSettings");
                break;
            case "project":
                result.push("projectSettings");
                break;
            case "local":
                result.push("localSettings");
                break;
            case "policy":
                result.push("policySettings");
                break;
            case "flag":
                result.push("flagSettings");
                break;
        }
    }
    return result;
}

/**
 * Returns all valid settings sources.
 */
export function getAllSettingSources(): SettingsSource[] {
    return SETTING_SOURCES;
}

/**
 * Checks if a string is a valid setting source.
 */
export function isValidSettingSource(source: string): source is SettingsSource {
    return SETTING_SOURCES.includes(source as SettingsSource);
}

// --- Utilities ---

/**
 * Formats bytes into a human-readable string (KB, MB, GB).
 */
export function formatBytes(bytes: number): string {
    const kb = bytes / 1024;
    if (kb < 1) return `${bytes} bytes`;
    if (kb < 1024) return `${kb.toFixed(1).replace(/\.0$/, "")}KB`;
    const mb = kb / 1024;
    if (mb < 1024) return `${mb.toFixed(1).replace(/\.0$/, "")}MB`;
    return `${(mb / 1024).toFixed(1).replace(/\.0$/, "")}GB`;
}

function writeAtomic(filePath: string, content: string) {
    const tempPath = `${filePath}.tmp.${process.pid}.${Date.now()}`;
    try {
        fs.writeFileSync(tempPath, content, { encoding: "utf-8", flush: true });

        if (fs.existsSync(filePath)) {
            const stats = fs.statSync(filePath);
            fs.chmodSync(tempPath, stats.mode);
        }

        fs.renameSync(tempPath, filePath);
    } catch (err) {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
        throw err;
    }
}

function deepMerge(target: any, source: any): any {
    const output = { ...target };
    if (isObject(target) && isObject(source)) {
        Object.keys(source).forEach((key) => {
            if (isObject(source[key])) {
                if (!(key in target)) {
                    Object.assign(output, { [key]: source[key] });
                } else {
                    output[key] = deepMerge(target[key], source[key]);
                }
            } else if (Array.isArray(source[key])) {
                // For arrays in permissions (allow, deny, ask), we often want to merge/union
                if (["allow", "deny", "ask", "additionalDirectories"].includes(key)) {
                    const targetArr = Array.isArray(target[key]) ? target[key] : [];
                    output[key] = Array.from(new Set([...targetArr, ...source[key]]));
                } else {
                    output[key] = source[key];
                }
            } else {
                Object.assign(output, { [key]: source[key] });
            }
        });
    }
    return output;
}

function isObject(item: any) {
    return item && typeof item === "object" && !Array.isArray(item);
}
