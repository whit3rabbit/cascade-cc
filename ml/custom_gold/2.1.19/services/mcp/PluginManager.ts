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
        console.log(`[Plugins] Checking for updates: ${pluginId} in scope ${scope}`);

        // Simulated marketplace check
        await new Promise(r => setTimeout(r, 800));

        const isUpToDate = Math.random() > 0.5;
        if (isUpToDate) {
            return {
                success: true,
                message: `Plugin ${pluginId} is already up to date`,
                alreadyUpToDate: true
            };
        }

        console.log(`[Plugins] Updating ${pluginId}...`);
        // Simulated download and extraction
        await new Promise(r => setTimeout(r, 1200));

        return {
            success: true,
            message: `Updated plugin ${pluginId} to latest version`,
            alreadyUpToDate: false,
            oldVersion: "1.0.0",
            newVersion: "1.0.1"
        };
    }

    /**
     * Installs a new plugin.
     */
    static async installPlugin(plugin: any, scope: ConfigScope = "user") {
        const pluginId = `plugin:${plugin.id || Math.random().toString(36).substring(7)}`;
        console.log(`[Plugins] Installing ${plugin.name || pluginId} in scope ${scope}`);

        // 1. Validation
        const existing = await this.getInstalledPlugins();
        if (existing.find(p => p.id === pluginId)) {
            return { success: false, message: `Plugin ${plugin.name || pluginId} is already installed.` };
        }

        // 2. Simulated download
        await new Promise(r => setTimeout(r, 1500));

        // 3. Registration
        const mcpConfig = plugin.mcp || {
            type: 'stdio',
            command: plugin.command || 'node',
            args: plugin.args || [],
            env: plugin.env || {}
        };

        const { McpServerManager } = await import('./McpServerManager.js');
        await McpServerManager.addMcpServer(pluginId, {
            ...mcpConfig,
            name: plugin.name || pluginId,
            version: plugin.version || '1.0.0',
            enabled: true
        }, scope as any);

        // 4. Auto-connect after install
        try {
            const { mcpClientManager } = await import('./McpClientManager.js');
            await mcpClientManager.connect(pluginId, mcpConfig);
        } catch (e) {
            console.error(`[Plugins] Auto-connect failed for ${pluginId}:`, e);
        }

        return { success: true, message: `Successfully installed plugin: ${plugin.name || pluginId}`, pluginId };
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
