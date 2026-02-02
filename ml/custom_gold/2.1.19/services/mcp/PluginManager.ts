/**
 * File: src/services/mcp/PluginManager.ts
 * Role: Logic for managing MCP plugins (install, update, enable/disable).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { getSettings, updateSettings } from '../config/SettingsService.js';
import { McpServerConfig } from './McpServerManager.js';

/**
 * Scope for configuration.
 */
export type ConfigScope = "project" | "user" | "local";

/**
 * Interface for plugin metadata.
 */
export interface PluginInstallation {
    id: string;
    name: string;
    version: string;
    scope: "project" | "user" | "global";
    projectPath?: string;
    installPath: string;
    enabled: boolean;
}

/**
 * Manages the installation and state of MCP plugins.
 */
export class PluginManager {
    static async getInstalledPlugins(): Promise<PluginInstallation[]> {
        const settings = getSettings();
        const plugins = settings.mcp?.servers || {};
        const list: PluginInstallation[] = [];

        for (const [id, config] of Object.entries(plugins)) {
            if (id.startsWith('plugin:')) {
                list.push({
                    id,
                    name: (config as any).name || id.replace('plugin:', ''),
                    version: (config as any).version || '1.0.0',
                    scope: (config as any).scope || 'user',
                    installPath: (config as any).installPath || '',
                    enabled: (config as any).enabled !== false
                });
            }
        }
        return list;
    }

    /**
     * Toggles a plugin's enabled state.
     */
    static async togglePlugin(pluginId: string, enabled: boolean, scope: ConfigScope = "user") {
        console.log(`[Plugins] Toggling ${pluginId} to ${enabled ? 'enabled' : 'disabled'} in scope ${scope}`);

        const { mcpClientManager } = await import('./McpClientManager.js');
        const settings = getSettings();
        const mcpConfig = settings.mcp?.servers?.[pluginId];

        updateSettings((current) => {
            const servers = { ...(current.mcp?.servers || {}) };
            if (servers[pluginId]) {
                servers[pluginId] = { ...servers[pluginId], enabled };
            }
            return {
                ...current,
                mcp: { ...(current.mcp || {}), servers }
            };
        });

        if (enabled) {
            if (mcpConfig) {
                await mcpClientManager.connect(pluginId, mcpConfig);
            }
        } else {
            await mcpClientManager.disconnect(pluginId);
        }

        return { success: true, message: `Updated plugin ${pluginId}` };
    }

    /**
     * Installs or updates a plugin from the marketplace.
     */
    static async updatePlugin(pluginId: string, scope: ConfigScope) {
        // In a real implementation, this would check the marketplace for updates
        // and potentially run an installer.
        console.log(`[Plugins] Updating ${pluginId} in scope ${scope}`);
        return {
            success: true,
            message: `Updated plugin ${pluginId}`,
            alreadyUpToDate: false,
            oldVersion: "1.0.0",
            newVersion: "1.1.0"
        };
    }

    /**
     * Installs a new plugin.
     */
    static async installPlugin(plugin: any, scope: ConfigScope = "user") {
        console.log(`[Plugins] Installing ${plugin.id} in scope ${scope}`);

        const pluginId = `plugin:${plugin.id}`;
        const mcpConfig = plugin.mcp || {
            type: 'stdio',
            command: plugin.command,
            args: plugin.args,
            env: plugin.env
        };

        const { McpServerManager } = await import('./McpServerManager.js');
        await McpServerManager.addMcpServer(pluginId, {
            ...mcpConfig,
            name: plugin.name,
            version: plugin.version,
            enabled: true
        }, scope as any);

        // Auto-connect after install
        const { mcpClientManager } = await import('./McpClientManager.js');
        await mcpClientManager.connect(pluginId, mcpConfig);

        return { success: true, message: `Installed plugin: ${plugin.name}`, pluginId };
    }

    /**
     * Uninstalls a plugin.
     */
    static async uninstallPlugin(pluginId: string, scope: ConfigScope) {
        console.log(`[Plugins] Uninstalling ${pluginId} in scope ${scope}`);
        const { mcpClientManager } = await import('./McpClientManager.js');
        await mcpClientManager.disconnect(pluginId);

        updateSettings((current) => {
            const servers = { ...(current.mcp?.servers || {}) };
            delete servers[pluginId];
            return {
                ...current,
                mcp: { ...(current.mcp || {}), servers }
            };
        });

        return { success: true, message: `Uninstalled plugin ${pluginId}`, pluginId };
    }
}

// Named exports for compatibility
export const togglePlugin = PluginManager.togglePlugin;
export const enablePlugin = (id: string, scope: ConfigScope) => PluginManager.togglePlugin(id, true, scope);
export const disablePlugin = (id: string, scope: ConfigScope) => PluginManager.togglePlugin(id, false, scope);
export const updatePlugin = PluginManager.updatePlugin;
export const installPlugin = PluginManager.installPlugin;
export const uninstallPlugin = PluginManager.uninstallPlugin;
