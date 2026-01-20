import { resolve } from "node:path";
import { getSettings, updateSettings, SettingsSource, SETTING_SOURCES, Settings } from "./settings.js";
import { log } from "../logger/loggerService.js";
import { checkStatsigGate } from "./GrowthBookService.js";

const logger = log("permissions");

// Types
export type PermissionMode = "default" | "bypassPermissions" | "prompt" | "acceptEdis";

export interface ToolPermissionContext {
    mode: PermissionMode;
    additionalWorkingDirectories: Map<string, any>;
    alwaysAllowRules: Record<string, string[]>;
    alwaysDenyRules: Record<string, string[]>;
    alwaysAskRules: Record<string, string[]>;
    isBypassPermissionsModeAvailable: boolean;
}

/**
 * Returns a fresh, default ToolPermissionContext. Deobfuscated from TN in chunk_217.ts.
 */
export function createDefaultPermissionContext(): ToolPermissionContext {
    return {
        mode: "default",
        additionalWorkingDirectories: new Map(),
        alwaysAllowRules: {},
        alwaysDenyRules: {},
        alwaysAskRules: {},
        isBypassPermissionsModeAvailable: false,
    };
}

/**
 * Resolves the active permission mode based on CLI arguments and settings.
 * Deobfuscated from f69 in chunk_532.ts.
 */
export function resolvePermissionMode({
    permissionModeCli,
    dangerouslySkipPermissions,
}: {
    permissionModeCli?: string;
    dangerouslySkipPermissions?: boolean;
}): { mode: PermissionMode; notification?: string } {
    const settings = getSettings("userSettings") || {};

    // Check if bypass mode is disabled by policy or settings
    const isBypassDisabledByGate = checkStatsigGate("tengu_disable_bypass_permissions_mode");
    const isBypassDisabledBySettings = settings.permissions?.disableBypassPermissionsMode === "disable";
    const isBypassForbidden = isBypassDisabledByGate || isBypassDisabledBySettings;

    const candidateModes: string[] = [];
    if (dangerouslySkipPermissions) candidateModes.push("bypassPermissions");
    if (permissionModeCli) candidateModes.push(permissionModeCli);
    if (settings.permissions?.defaultMode) candidateModes.push(settings.permissions.defaultMode);

    let notification: string | undefined;

    for (const mode of candidateModes) {
        if (mode === "bypassPermissions" && isBypassForbidden) {
            if (isBypassDisabledByGate) {
                logger.warn("bypassPermissions mode is disabled by organization policy");
                notification = "Bypass permissions mode was disabled by your organization policy";
            } else {
                logger.warn("bypassPermissions mode is disabled by settings");
                notification = "Bypass permissions mode was disabled by settings";
            }
            continue;
        }
        return { mode: mode as PermissionMode, notification };
    }

    return { mode: "default", notification };
}

/**
 * Extracts permission rules from a settings object. Deobfuscated from g23 in chunk_218.ts.
 */
export function extractPermissionsFromSettings(settings: Settings, source: SettingsSource) {
    if (!settings || !settings.permissions) return [];
    const { permissions } = settings;
    const rules: any[] = [];
    const behaviors: ("allow" | "deny" | "ask")[] = ["allow", "deny", "ask"];

    for (const behavior of behaviors) {
        const values = permissions[behavior];
        if (values && Array.isArray(values)) {
            for (const value of values) {
                rules.push({
                    source,
                    ruleBehavior: behavior,
                    ruleValue: value // Should normalize value? gM(J)
                });
            }
        }
    }
    return rules;
}

/**
 * Aggregates permissions from all persistent sources. Deobfuscated from MA1 in chunk_218.ts.
 */
export function getAllPersistedPermissions() {
    const allRules: any[] = [];
    for (const source of SETTING_SOURCES) {
        allRules.push(...getPersistedPermissionsForSource(source));
    }
    return allRules;
}

/**
 * Gets permissions for a specific source. Deobfuscated from RA1 in chunk_218.ts.
 */
export function getPersistedPermissionsForSource(source: SettingsSource) {
    const settings = getSettings(source);
    return extractPermissionsFromSettings(settings, source);
}

/**
 * Deletes a specific rule from persistent settings. Deobfuscated from gFB in chunk_218.ts.
 */
export function removePersistedPermission(rule: { source: SettingsSource, ruleBehavior: string, ruleValue: string }) {
    const mutableSources: SettingsSource[] = ["userSettings", "projectSettings", "localSettings"];
    if (!mutableSources.includes(rule.source)) return false;

    const settings = getSettings(rule.source);
    if (!settings || !settings.permissions) return false;

    const behavior = rule.ruleBehavior as "allow" | "deny" | "ask";
    const currentRules = settings.permissions[behavior];
    if (!currentRules || !currentRules.includes(rule.ruleValue)) return false;

    const updatedRules = currentRules.filter(v => v !== rule.ruleValue);
    const updatedSettings = {
        ...settings,
        permissions: {
            ...settings.permissions,
            [behavior]: updatedRules
        }
    };

    const { error } = updateSettings(rule.source, updatedSettings);
    return !error;
}

/**
 * Saves multiple rules to a settings source. Deobfuscated from _A1 in chunk_218.ts.
 */
export function persistPermissions(data: { ruleValues: string[], ruleBehavior: "allow" | "deny" | "ask" }, source: SettingsSource) {
    if (data.ruleValues.length < 1) return true;

    const settings = getSettings(source);
    const permissions = settings.permissions || {};
    const currentRules = permissions[data.ruleBehavior] || [];
    const ruleSet = new Set(currentRules);

    const newRules = data.ruleValues.filter(v => !ruleSet.has(v));
    if (newRules.length === 0) return true;

    const updatedSettings = {
        ...settings,
        permissions: {
            ...permissions,
            [data.ruleBehavior]: [...currentRules, ...newRules]
        }
    };

    const { error } = updateSettings(source, updatedSettings);
    return !error;
}
