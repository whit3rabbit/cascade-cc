import * as path from 'node:path';

export function parsePluginId(id: string): { name: string, marketplace?: string } {
    if (id.includes('@')) {
        const [name, marketplace] = id.split('@');
        return { name, marketplace };
    }
    return { name: id };
}

export function validateScope(scope: string): void {
    const validScopes = ["user", "project", "local", "managed"];
    if (!validScopes.includes(scope)) {
        throw new Error(`Invalid scope: ${scope}`);
    }
}

export function getProjectPathForScope(scope: string): string | undefined {
    if (scope === "project" || scope === "local") {
        return process.cwd();
    }
    return undefined;
}

export function isLocalSource(source: string): boolean {
    return source.startsWith('/') || source.startsWith('./') || source.startsWith('../');
}

import { SettingsSource } from '../terminal/settings.js';

export function getSettingsScope(scope: string): SettingsSource {
    switch (scope) {
        case "user": return "userSettings";
        case "project": return "projectSettings";
        case "local": return "localSettings";
        case "managed": return "policySettings";
        default: return "userSettings";
    }
}
