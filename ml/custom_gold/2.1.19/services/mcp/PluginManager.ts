
import * as fs from 'node:fs';
import * as path from 'node:path';
import { getSettings, updateSettings } from '../config/SettingsService.js';
import { gitClone, gitPull } from '../../utils/shared/git.js';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';
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
    private static getPluginsDir(): string {
        const configDir = getBaseConfigDir();
        const pluginsDir = path.join(configDir, 'plugins');
        if (!fs.existsSync(pluginsDir)) {
            fs.mkdirSync(pluginsDir, { recursive: true });
        }
        return pluginsDir;
    }

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

        const settings = getSettings();
        const config = settings.mcp?.servers?.[pluginId];

        if (!config || !(config as any).installPath) {
            return { success: false, message: `Plugin ${pluginId} not found or has no install path` };
        }

        const installPath = (config as any).installPath;
        if (!fs.existsSync(installPath)) {
            return { success: false, message: `Plugin directory not found at ${installPath}` };
        }

        try {
            // Perform pull and capture output
            const { execSync } = await import('child_process');
            const pullOutput = execSync('git pull', {
                cwd: installPath,
                encoding: 'utf8',
                stdio: ['ignore', 'pipe', 'ignore']
            });

            const alreadyUpToDate = pullOutput.includes('Already up to date');
            console.log(`[Plugins] Git pull output: ${pullOutput.trim()}`);

            // Re-install dependencies after pull
            const packageJsonPath = path.join(installPath, 'package.json');
            if (fs.existsSync(packageJsonPath)) {
                const { execSync } = await import('child_process');
                execSync('npm install --no-audit --no-fund --omit=dev', {
                    cwd: installPath,
                    stdio: 'inherit'
                });
            }

            // Verify if restart is needed? For now we just pull.
            return {
                success: true,
                message: alreadyUpToDate ? `Plugin ${pluginId} is already up to date` : `Updated plugin ${pluginId} to latest version`,
                alreadyUpToDate,
                // In a real implementation we would check git status/log for versions
                oldVersion: "unknown",
                newVersion: "latest"
            };
        } catch (e) {
            return { success: false, message: `Failed to update plugin: ${(e as Error).message}` };
        }
    }

    /**
     * Installs a new plugin.
     */
    static async installPlugin(plugin: any, scope: ConfigScope = "user") {
        // Handle git source
        if (plugin.source === 'github' && plugin.repository) {
            const repoName = plugin.repository.split('/').pop();
            const pluginId = `plugin:${repoName}`;

            const existing = await this.getInstalledPlugins();
            if (existing.find(p => p.id === pluginId)) {
                return { success: false, message: `Plugin ${plugin.name || pluginId} is already installed.` };
            }

            const installPath = path.join(this.getPluginsDir(), repoName);
            const repoUrl = `https://github.com/${plugin.repository}.git`;

            console.log(`[Plugins] Cloning ${repoUrl} to ${installPath}`);
            try {
                if (fs.existsSync(installPath)) {
                    // Look for existing install but maybe not in config?
                    // Trashing existing dir to be safe or just pulling?
                    // Safest is to error or pull. Let's pull if exists.
                    await gitPull(installPath);
                } else {
                    await gitClone(repoUrl, installPath);
                }
            } catch (e) {
                return { success: false, message: `Failed to clone repository: ${(e as Error).message}` };
            }

            // 2b. Install dependencies
            const packageJsonPath = path.join(installPath, 'package.json');
            if (fs.existsSync(packageJsonPath)) {
                try {
                    console.log(`[Plugins] Installing dependencies for ${plugin.name || pluginId}...`);
                    // Use --no-audit --no-fund --omit=dev for a cleaner, production-like install
                    const { execSync } = await import('child_process');
                    execSync('npm install --no-audit --no-fund --omit=dev', {
                        cwd: installPath,
                        stdio: 'inherit' // Show output to user
                    });
                } catch (e) {
                    console.error(`[Plugins] Failed to install dependencies: ${(e as Error).message}`);
                    return { success: false, message: `Failed to install dependencies: ${(e as Error).message}` };
                }
            }

            // 3. Registration
            // We need to look for a claude-plugin.json or similar manifest in the cloned dir?
            // Or assumes stdio/node structure.
            // For now, let's assume default node structure if not provided in `plugin` object.

            let mcpConfig = plugin.mcp;
            if (!mcpConfig) {
                // Try to detect
                const packageJsonPath = path.join(installPath, 'package.json');
                if (fs.existsSync(packageJsonPath)) {
                    try {
                        const pkg = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
                        // If package.json has "bin", use it?
                        // Or assume `npm install && npm start`?
                        // The reference implementation might run `npm install` here.
                        // We already ran npm install above.


                        mcpConfig = {
                            type: 'stdio',
                            command: 'node',
                            args: [path.join(installPath, pkg.main || 'index.js')],
                            env: plugin.env || {}
                        };
                    } catch (e) {
                        console.error('Failed to parse package.json', e);
                    }
                }
            }

            if (!mcpConfig) {
                mcpConfig = {
                    type: 'stdio',
                    command: 'node',
                    args: [path.join(installPath, 'index.js')],
                    env: plugin.env || {}
                };
            }

            const { McpServerManager } = await import('./McpServerManager.js');
            await McpServerManager.addMcpServer(pluginId, {
                ...mcpConfig,
                name: plugin.name || repoName,
                version: plugin.version || '1.0.0',
                enabled: true,
                installPath, // Store install path for updates
                source: plugin.source,
                repository: plugin.repository
            } as any, scope as any);

            // 4. Auto-connect after install
            try {
                const { mcpClientManager } = await import('./McpClientManager.js');
                await mcpClientManager.connect(pluginId, mcpConfig);
            } catch (e) {
                console.error(`[Plugins] Auto-connect failed for ${pluginId}:`, e);
            }

            return { success: true, message: `Successfully installed plugin: ${plugin.name || pluginId}`, pluginId };
        } else {
            return { success: false, message: `Unsupported plugin source: ${plugin.source}` };
        }
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
            // If we want to delete file, checking installPath would be good.
            // But we might want to keep data?
            // User can manually remove.
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
