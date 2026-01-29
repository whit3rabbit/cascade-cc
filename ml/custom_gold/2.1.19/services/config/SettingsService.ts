/**
 * File: src/services/config/SettingsService.ts
 * Role: Persistent settings management for user and project-specific configurations.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';

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
        };
    };
    telemetry?: { enabled: boolean };
    theme?: "dark" | "light" | "system";
    iterm2It2SetupComplete?: boolean;
    preferTmuxOverIterm2?: boolean;
    toolSettings?: any;
    [key: string]: any;
}

export function getToolSettings(layer: string): any {
    // Stub implementation to satisfy SandboxSettings.ts
    // In a real app this would look up settings by layer
    return {};
}

let cachedSettings: Settings | null = null;

/**
 * Retrieves all persistent settings.
 */
export function getSettings(): Settings {
    if (cachedSettings) return cachedSettings;

    const settingsPath = join(getBaseConfigDir(), 'settings.json');
    if (!existsSync(settingsPath)) {
        return {
            sandbox: { enabled: false },
            telemetry: { enabled: true },
            theme: "dark"
        };
    }

    try {
        const data = readFileSync(settingsPath, 'utf8');
        cachedSettings = JSON.parse(data);
        return cachedSettings || {};
    } catch (error) {
        console.error("[Settings] Failed to read settings.json:", error);
        return {};
    }
}

/**
 * Updates settings and persists them to disk.
 * Supports partial updates or function-based updates.
 */
export function updateSettings(updates: Partial<Settings> | ((current: Settings) => Settings)): void {
    const current = getSettings();
    const updated = typeof updates === 'function' ? updates(current) : { ...current, ...updates };

    const settingsPath = join(getBaseConfigDir(), 'settings.json');
    const configDir = dirname(settingsPath);

    try {
        if (!existsSync(configDir)) mkdirSync(configDir, { recursive: true });
        writeFileSync(settingsPath, JSON.stringify(updated, null, 2), 'utf8');
        cachedSettings = updated;
    } catch (error) {
        console.error("[Settings] Failed to save settings.json:", error);
    }
}

/**
 * Initializes settings on app startup.
 */
export async function initializeSettings() {
    return getSettings();
}
