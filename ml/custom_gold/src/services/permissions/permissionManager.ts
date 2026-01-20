
import {
    getSettings,
    updateSettings
} from "../terminal/settings.js";

/**
 * Permission types and reducer.
 * Deobfuscated from WW, li, $v, etc. in chunk_219.ts.
 */

export type PermissionMode = "allow" | "deny" | "ask";
export type PermissionBehavior = "allow" | "deny" | "ask";
export type PermissionDestination = "localSettings" | "userSettings" | "projectSettings" | "session";

export interface PermissionRule {
    toolName: string;
    ruleContent: string;
}

export interface PermissionState {
    mode: PermissionMode;
    alwaysAllowRules: Record<string, string[]>;
    alwaysDenyRules: Record<string, string[]>;
    alwaysAskRules: Record<string, string[]>;
    additionalWorkingDirectories: Map<string, { path: string, source: string }>;
}

export type PermissionAction =
    | { type: "setMode"; mode: PermissionMode; destination: PermissionDestination }
    | { type: "addRules"; behavior: PermissionBehavior; rules: PermissionRule[]; destination: PermissionDestination }
    | { type: "replaceRules"; behavior: PermissionBehavior; rules: PermissionRule[]; destination: PermissionDestination }
    | { type: "removeRules"; behavior: PermissionBehavior; rules: PermissionRule[]; destination: PermissionDestination }
    | { type: "addDirectories"; directories: string[]; destination: PermissionDestination }
    | { type: "removeDirectories"; directories: string[]; destination: PermissionDestination };

/**
 * Normalizes a rule for serialization/comparison.
 */
function normalizeRule(rule: PermissionRule): string {
    return `${rule.toolName}:${rule.ruleContent}`;
}

/**
 * Core reducer for permission state.
 */
export function permissionReducer(state: PermissionState, action: PermissionAction): PermissionState {
    switch (action.type) {
        case "setMode":
            return { ...state, mode: action.mode };

        case "addRules": {
            const normalized = action.rules.map(normalizeRule);
            const key = action.behavior === "allow" ? "alwaysAllowRules" :
                action.behavior === "deny" ? "alwaysDenyRules" : "alwaysAskRules";
            return {
                ...state,
                [key]: {
                    ...state[key],
                    [action.destination]: [...(state[key][action.destination] || []), ...normalized]
                }
            };
        }

        case "replaceRules": {
            const normalized = action.rules.map(normalizeRule);
            const key = action.behavior === "allow" ? "alwaysAllowRules" :
                action.behavior === "deny" ? "alwaysDenyRules" : "alwaysAskRules";
            return {
                ...state,
                [key]: {
                    ...state[key],
                    [action.destination]: normalized
                }
            };
        }

        case "removeRules": {
            const normalized = action.rules.map(normalizeRule);
            const key = action.behavior === "allow" ? "alwaysAllowRules" :
                action.behavior === "deny" ? "alwaysDenyRules" : "alwaysAskRules";
            const existing = state[key][action.destination] || [];
            const toRemove = new Set(normalized);
            return {
                ...state,
                [key]: {
                    ...state[key],
                    [action.destination]: existing.filter(r => !toRemove.has(r))
                }
            };
        }

        case "addDirectories": {
            const nextDirs = new Map(state.additionalWorkingDirectories);
            for (const dir of action.directories) {
                nextDirs.set(dir, { path: dir, source: action.destination });
            }
            return { ...state, additionalWorkingDirectories: nextDirs };
        }

        case "removeDirectories": {
            const nextDirs = new Map(state.additionalWorkingDirectories);
            for (const dir of action.directories) {
                nextDirs.delete(dir);
            }
            return { ...state, additionalWorkingDirectories: nextDirs };
        }

        default:
            return state;
    }
}

/**
 * Applies a batch of actions to state.
 */
export function applyPermissionUpdates(state: PermissionState, actions: PermissionAction[]): PermissionState {
    return actions.reduce(permissionReducer, state);
}

/**
 * Persists a single update to settings on disk.
 */
export function persistPermissionUpdate(action: PermissionAction) {
    const dest = action.destination;
    if (dest === "session") return;

    const current = getSettings(dest);
    const permissions = current?.permissions || {};

    switch (action.type) {
        case "setMode":
            updateSettings(dest, {
                permissions: { ...permissions, defaultMode: action.mode as any }
            });
            break;

        case "addRules": {
            const existing = permissions[action.behavior] || [];
            const normalized = action.rules.map(normalizeRule);
            updateSettings(dest, {
                permissions: { ...permissions, [action.behavior]: [...existing, ...normalized] }
            });
            break;
        }

        case "replaceRules": {
            const normalized = action.rules.map(normalizeRule);
            updateSettings(dest, {
                permissions: { ...permissions, [action.behavior]: normalized }
            });
            break;
        }

        case "removeRules": {
            const existing = permissions[action.behavior] || [];
            const normalizedToRemove = new Set(action.rules.map(normalizeRule));
            const newRules = existing.filter((r: string) => !normalizedToRemove.has(r));
            updateSettings(dest, {
                permissions: { ...permissions, [action.behavior]: newRules }
            });
            break;
        }

        case "addDirectories": {
            const existing = permissions.additionalDirectories || [];
            updateSettings(dest, {
                permissions: {
                    ...permissions,
                    additionalDirectories: Array.from(new Set([...existing, ...action.directories]))
                }
            });
            break;
        }

        case "removeDirectories": {
            const existing = permissions.additionalDirectories || [];
            const toRemove = new Set(action.directories);
            const newDirs = existing.filter((d: string) => !toRemove.has(d));
            updateSettings(dest, {
                permissions: {
                    ...permissions,
                    additionalDirectories: newDirs
                }
            });
            break;
        }
    }
}

/**
 * Persists a batch of updates.
 */
export function persistPermissionUpdates(actions: PermissionAction[]) {
    for (const action of actions) {
        persistPermissionUpdate(action);
    }
}
