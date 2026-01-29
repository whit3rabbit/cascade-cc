/**
 * File: src/services/plugin/LspManager.ts
 * Role: Manages LSP (Language Server Protocol) configurations for plugins.
 */

import { getMissingVariables } from '../../utils/shared/commandStringProcessing.js';

export interface LspConfig {
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    [key: string]: any;
}

export interface ProcessingResult {
    config: LspConfig;
    missingVars: string[];
}

export interface Plugin {
    name: string;
    enabled: boolean;
    path: string;
    lspServers?: Record<string, LspConfig>;
}

/**
 * Replaces ${CLAUDE_PLUGIN_ROOT} placeholder with the actual plugin path.
 */
function expandPluginPath(str: string, pluginPath: string): string {
    return str.replace(/\$\{CLAUDE_PLUGIN_ROOT\}/g, pluginPath);
}

/**
 * Processes an LSP configuration, expanding environment variables and placeholders.
 */
export function processLspConfig(config: LspConfig, pluginPath: string): ProcessingResult {
    const updated: LspConfig = { ...config };
    const missingVars: string[] = [];

    const expand = (val: string): string => {
        const withPath = expandPluginPath(val, pluginPath);
        const { expanded, missingVars: newVars } = getMissingVariables(withPath);
        missingVars.push(...newVars);
        return expanded;
    };

    if (updated.command) {
        updated.command = expand(updated.command);
    }

    if (updated.args) {
        updated.args = updated.args.map(expand);
    }

    if (updated.env) {
        const newEnv: Record<string, string> = { CLAUDE_PLUGIN_ROOT: pluginPath };
        for (const [k, v] of Object.entries(updated.env)) {
            newEnv[k] = expand(v);
        }
        updated.env = newEnv;
    }

    return {
        config: updated,
        missingVars: [...new Set(missingVars)]
    };
}

/**
 * Loads LSP server configurations for a specific enabled plugin.
 */
export async function loadPluginLspServers(plugin: Plugin): Promise<Record<string, any>> {
    if (!plugin.enabled || !plugin.lspServers) {
        return {};
    }

    const servers: Record<string, any> = {};
    for (const [key, rawConfig] of Object.entries(plugin.lspServers)) {
        const { config, missingVars } = processLspConfig(rawConfig, plugin.path);

        if (missingVars.length > 0) {
            console.warn(`Plugin ${plugin.name} is missing env vars for ${key}: ${missingVars.join(', ')}`);
        }

        const serverId = `plugin:${plugin.name}:${key}`;
        servers[serverId] = {
            ...config,
            scope: "dynamic",
            source: plugin.name
        };
    }
    return servers;
}
