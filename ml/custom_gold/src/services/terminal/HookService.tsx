
// Logic from chunk_584.ts (Permission GUI & Hook Management)

import React, { useCallback } from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../../components/permissions/PermissionComponents.js";
import { getSettings, updateSettings } from "./settings.js";

// --- Permission Source Labels ---
export function getFriendlySourceName(source: string): string {
    switch (source) {
        case "userSettings": return "User Settings (~/.claude/settings.json)";
        case "projectSettings": return "Project Settings (.claude/settings.json)";
        case "localSettings": return "Local Settings (.claude/settings.local.json)";
        case "session": return "Session Cache (In-memory)";
        default: return source;
    }
}

// --- Add Permission Rule View (T89) ---
export function AddPermissionRuleView({
    rules,
    behavior,
    onAdd,
    onCancel
}: any) {
    const options = [
        { label: "Local Settings", value: "localSettings" },
        { label: "Project Settings", value: "projectSettings" },
        { label: "User Settings", value: "userSettings" }
    ];

    const handleSubmit = useCallback((source: string) => {
        if (source === "cancel") {
            onCancel();
            return;
        }
        onAdd(rules, behavior, source);
    }, [onAdd, onCancel, rules, behavior]);

    const title = `Add ${behavior} permission rule${rules.length === 1 ? "" : "s"}`;

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="permission" paddingX={1}>
            <Text bold color="permission">{title}</Text>
            <Box flexDirection="column" marginY={1}>
                {rules.map((rule: any, i: number) => (
                    <Text key={i}>{rule.toolName || JSON.stringify(rule)}</Text>
                ))}
            </Box>
            <Text>Where should this rule be saved?</Text>
            <PermissionSelect
                options={[...options, { label: "Cancel", value: "cancel" }]}
                onChange={handleSubmit}
                onCancel={onCancel}
            />
        </Box>
    );
}

// --- Hook Management (DJ9, lY9, CJ9) ---

export interface HookConfig {
    event: string;
    matcher: string;
    hooks: any[];
    source: string;
}

export function getAllHooks(settings: any): HookConfig[] {
    const hooks: HookConfig[] = [];
    const sources = ["userSettings", "projectSettings", "localSettings"];

    for (const source of sources) {
        const sourceHooks = settings[source]?.hooks || {};
        for (const [event, configs] of Object.entries(sourceHooks)) {
            (configs as any[]).forEach((cfg: any) => {
                hooks.push({
                    event,
                    matcher: cfg.matcher,
                    hooks: cfg.hooks,
                    source
                });
            });
        }
    }
    return hooks;
}

export type HookEntry = {
    event: string;
    config: any;
    matcher?: string;
    source: string;
    pluginName?: string;
};

function normalizeMatcher(matcher?: string) {
    return matcher ?? "";
}

export async function addHookToSettings(event: string, config: any, matcher: string, source: string) {
    const settings = getSettings(source as any) ?? {};
    const hooks = { ...(settings as any).hooks } as Record<string, any[]>;
    const normalizedMatcher = normalizeMatcher(matcher);
    const eventConfigs = [...(hooks[event] ?? [])];

    let entry = eventConfigs.find((item) => normalizeMatcher(item.matcher) === normalizedMatcher);
    if (!entry) {
        entry = { matcher: normalizedMatcher, hooks: [] };
        eventConfigs.push(entry);
    }

    entry.hooks = [...(entry.hooks ?? []), config];
    hooks[event] = eventConfigs;

    updateSettings(source as any, { hooks });
}

export async function removeHookFromSettings(hook: HookEntry) {
    const settings = getSettings(hook.source as any) ?? {};
    const hooks = { ...(settings as any).hooks } as Record<string, any[]>;
    const normalizedMatcher = normalizeMatcher(hook.matcher);
    const eventConfigs = [...(hooks[hook.event] ?? [])];

    const nextEventConfigs = eventConfigs
        .map((entry) => {
            if (normalizeMatcher(entry.matcher) !== normalizedMatcher) return entry;
            const nextHooks = (entry.hooks ?? []).filter(
                (candidate: any) => JSON.stringify(candidate) !== JSON.stringify(hook.config)
            );
            return { ...entry, hooks: nextHooks };
        })
        .filter((entry) => (entry.hooks ?? []).length > 0);

    if (nextEventConfigs.length === 0) {
        delete hooks[hook.event];
    } else {
        hooks[hook.event] = nextEventConfigs;
    }

    updateSettings(hook.source as any, { hooks });
}

/**
 * Returns a human-readable list of changes in hooks configuration.
 */
export function diffHooks(oldHooks: any, newHooks: any): string[] {
    const changes: string[] = [];
    if (JSON.stringify(oldHooks) !== JSON.stringify(newHooks)) {
        changes.push("Hooks configuration has been modified.");
    }
    return changes;
}
