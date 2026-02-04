/**
 * File: src/services/config/SettingsService.ts
 * Role: Persistent settings management with source-based layering (Policy -> Flag -> User -> Project -> Local).
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { homedir } from 'node:os';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';
import { EnvService } from './EnvService.js';
import { HooksConfig } from '../hooks/HookTypes.js';

export type SettingSource = "policySettings" | "flagSettings" | "legacySettings" | "userSettings" | "projectSettings" | "localSettings";

export interface ToolPermissionSettings {
    permissions?: {
        allow?: string[];
        deny?: string[];
        ask?: string[];
    };
    [key: string]: any;
}

export interface Settings {
    sandbox?: {
        enabled: boolean;
        ripgrep?: {
            command: string;
            args: string[];
        };
        network?: {
            allowUnixSockets?: boolean;
            allowAllUnixSockets?: boolean;
            allowLocalBinding?: boolean;
            allowedDomains?: string[];
            deniedDomains?: string[];
        };
        autoAllowBashIfSandboxed?: boolean;
        ignoreViolations?: boolean;
        excludedCommands?: string[];
        allowUnsandboxedCommands?: boolean;
    };
    allowManagedHooksOnly?: boolean;
    telemetry?: {
        enabled: boolean;
        enhanced?: boolean;
    };
    vimModeEnabled?: boolean;
    theme?: "dark" | "light" | "system" | "light-daltonized" | "dark-daltonized" | "light-ansi" | "dark-ansi";
    iterm2It2SetupComplete?: boolean;
    preferTmuxOverIterm2?: boolean;
    onboardingComplete?: boolean;
    numStartups?: number;
    tipsHistory?: Record<string, number>;
    lastPlanModeUse?: number;
    memoryUsageCount?: number;
    githubActionSetupCount?: number;
    slackAppInstallCount?: number;
    hasVisitedPasses?: boolean;
    subscriptionNoticeCount?: number;
    hasAvailableSubscription?: boolean;
    plansDirectory?: string;
    toolSettings?: {
        [layer: string]: ToolPermissionSettings;
    };
    hooks?: HooksConfig;
    autoCompact?: boolean;
    showTips?: boolean;
    thinkingMode?: boolean;
    promptSuggestions?: boolean;
    rewindCode?: boolean;
    verbose?: boolean;
    progressBar?: boolean;
    gitignore?: boolean;
    showCodeDiffFooter?: boolean;
    showPrStatusFooter?: boolean;
    autoConnectIde?: boolean;
    chromeEnabled?: boolean;
    repoPaths?: Record<string, string[]>;
    [key: string]: any;
}

const SOURCE_ORDER: SettingSource[] = [
    "policySettings",
    "flagSettings",
    "legacySettings",
    "userSettings",
    "projectSettings",
    "localSettings"
];

let cachedSettings: Settings | null = null;
let sourceCache: Map<SettingSource, Settings> = new Map();

/**
 * Returns the file path for a given settings source.
 */
export function getSettingsPath(source: SettingSource): string | null {
    const home = getBaseConfigDir();
    const cwd = process.cwd();

    switch (source) {
        case "userSettings":
            return join(home, 'settings.json');
        case "legacySettings":
            return join(homedir(), '.claude.json');
        case "projectSettings":
            return join(cwd, '.claude', 'settings.json');
        case "localSettings":
            return join(cwd, '.claude', 'settings.local.json');
        case "policySettings":
            if (process.platform === 'win32') return 'C:\\ProgramData\\ClaudeCode\\managed-settings.json';
            if (process.platform === 'darwin') return '/Library/Application Support/ClaudeCode/managed-settings.json';
            return '/etc/anthropic/claude/managed-settings.json';
        case "flagSettings":
            return EnvService.get('CLAUDE_FLAG_SETTINGS_PATH') || null;
        default:
            return null;
    }
}

/**
 * Deeply merges two objects.
 */
function deepMerge(target: any, source: any): any {
    if (!source) return target;
    if (!target) return source;

    const output = { ...target };
    if (typeof target === 'object' && typeof source === 'object' && !Array.isArray(target) && !Array.isArray(source)) {
        Object.keys(source).forEach(key => {
            if (typeof source[key] === 'object' && !Array.isArray(source[key]) && source[key] !== null) {
                if (!(key in target)) {
                    Object.assign(output, { [key]: source[key] });
                } else {
                    output[key] = deepMerge(target[key], source[key]);
                }
            } else {
                Object.assign(output, { [key]: source[key] });
            }
        });
    }
    return output;
}

/**
 * Loads settings for a specific source.
 */
export function loadSettingsForSource(source: SettingSource): Settings | null {
    const path = getSettingsPath(source);
    if (!path || !existsSync(path)) return null;

    try {
        const data = readFileSync(path, 'utf8');
        if (!data.trim()) return {};
        const parsed = JSON.parse(data);
        return parsed && typeof parsed === 'object' ? parsed : null;
    } catch (error) {
        console.error(`[Settings] Failed to read ${source} from ${path}:`, error);
        return null;
    }
}

/**
 * Retrieves all persistent settings, layered and merged.
 */
export function getSettings(): Settings {
    if (cachedSettings) return cachedSettings;

    let merged: Settings = {
        sandbox: { enabled: false },
        telemetry: { enabled: true },
        theme: "dark"
    };

    for (const source of SOURCE_ORDER) {
        const sourceSettings = loadSettingsForSource(source);
        if (sourceSettings) {
            merged = deepMerge(merged, sourceSettings);
            sourceCache.set(source, sourceSettings);
        }
    }

    cachedSettings = merged;
    return merged;
}

/**
 * Updates settings for a specific source and persists them to disk.
 */
export function updateSettingsForSource(source: SettingSource, updates: Partial<Settings> | ((current: Settings) => Settings)): void {
    const current = loadSettingsForSource(source) || {};
    const updated = typeof updates === 'function' ? updates(current) : deepMerge(current, updates);

    const path = getSettingsPath(source);
    if (!path) {
        console.error(`[Settings] No path found for source: ${source}`);
        return;
    }

    try {
        const configDir = dirname(path);
        if (!existsSync(configDir)) mkdirSync(configDir, { recursive: true });
        writeFileSync(path, JSON.stringify(updated, null, 2), 'utf8');

        // Invalidate caches
        sourceCache.set(source, updated);
        cachedSettings = null;
    } catch (error) {
        console.error(`[Settings] Failed to save ${source} to ${path}:`, error);
    }
}

/**
 * Legacy updateSettings - defaults to userSettings.
 */
export function updateSettings(updates: Partial<Settings> | ((current: Settings) => Settings)): void {
    updateSettingsForSource("userSettings", updates);
}

/**
 * Retrieves tool settings for a given layer.
 */
export function getToolSettings(layer: string): ToolPermissionSettings {
    const settings = getSettings();
    return settings.toolSettings?.[layer] || {};
}

/**
 * Updates tool settings for a given layer.
 */
export function setToolSettings(layer: string, settings: ToolPermissionSettings) {
    updateSettings(current => ({
        ...current,
        toolSettings: {
            ...current.toolSettings,
            [layer]: settings
        }
    }));
}



/**
 * Retrieves the directory for storing plans.
 * Defaults to .claude/plans in the current working directory.
 * Ensures the configured path is within the project root.
 */
export function getPlansDirectory(): string {
    const settings = getSettings();
    const projectRoot = process.cwd();
    const defaultPlansDir = join(projectRoot, '.claude', 'plans');

    if (settings.plansDirectory) {
        const resolved = resolve(projectRoot, settings.plansDirectory);
        if (!resolved.startsWith(projectRoot)) {
            console.error(`plansDirectory must be within project root: ${settings.plansDirectory}`);
            return defaultPlansDir;
        }
        return resolved;
    }
    return defaultPlansDir;
}

/**
 * Initializes settings on app startup.
 */
export async function initializeSettings() {
    return getSettings();
}
