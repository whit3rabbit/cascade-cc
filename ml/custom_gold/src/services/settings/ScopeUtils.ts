
export type Scope = 'user' | 'project' | 'local' | 'managed';
export type SettingsKey = 'userSettings' | 'projectSettings' | 'localSettings' | 'policySettings' | 'flagSettings';

export const SCOPE_TO_SETTINGS_KEY: Record<string, SettingsKey> = {
    user: "userSettings",
    project: "projectSettings",
    local: "localSettings"
};

export const SETTINGS_KEY_TO_SCOPE: Record<string, Scope> = {
    policySettings: "managed",
    userSettings: "user",
    projectSettings: "project",
    localSettings: "local"
};

export function getScopeSettingKey(scope: Scope): SettingsKey {
    if (scope === 'managed') throw new Error("Cannot install plugins to managed scope");
    return SCOPE_TO_SETTINGS_KEY[scope];
}

export function getScopeFromSettingsKey(key: SettingsKey): Scope {
    return SETTINGS_KEY_TO_SCOPE[key];
}
