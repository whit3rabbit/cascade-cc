
import * as fs from 'node:fs';
import * as path from 'node:path';
import { getSettings, updateSettings } from '../config/SettingsService.js';
import { gitClone, gitPull } from '../../utils/shared/git.js';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';
import semver from 'semver';
import { McpServerConfig } from './McpServerManager.js';
import { LspServerManager, LspServerConfig } from '../lsp/LspServerManager.js';
import { commandRegistry } from '../terminal/CommandRegistry.js';
import { registerAgent, loadAgent } from '../agents/AgentPersistence.js';
import { hookService } from '../hooks/HookService.js';
import matter from 'gray-matter';
import { terminalLog } from '../../utils/shared/runtime.js';

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
    /**
     * Initializes the plugin system, loading installed plugins and those specified via CLI.
     */
    static async initialize(options: { pluginDirs?: string[] }) {
        // 1. Load installed plugins
        try {
            const installedPlugins = await this.getInstalledPlugins();
            for (const p of installedPlugins) {
                if (p.enabled && p.installPath) {
                    await this.loadPluginFromDirectory(p.installPath, p.id);
                }
            }
        } catch (e) {
            terminalLog(`Failed to load installed plugins: ${e}`, "warn");
        }

        // 2. Load ephemeral plugins from CLI
        if (options.pluginDirs) {
            for (const dir of options.pluginDirs) {
                try {
                    await this.loadPluginFromDirectory(path.resolve(process.cwd(), dir));
                } catch (e) {
                    terminalLog(`Failed to load plugin from ${dir}: ${e}`, "error");
                }
            }
        }
    }

    /**
     * Loads a plugin from a directory.
     */
    static async loadPluginFromDirectory(pluginPath: string, existingId?: string) {
        if (!fs.existsSync(pluginPath)) {
            terminalLog(`Plugin directory not found: ${pluginPath}`, "warn");
            return;
        }

        // 1. Read Manifest
        let manifest: any = {};
        const manifestPath = path.join(pluginPath, '.claude-plugin', 'plugin.json');
        if (fs.existsSync(manifestPath)) {
            try {
                manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
            } catch (e) {
                terminalLog(`Failed to parse plugin manifest at ${manifestPath}: ${e}`, "error");
            }
        }

        const pluginName = manifest.name || existingId?.replace('plugin:', '') || path.basename(pluginPath);
        const namespace = pluginName;

        // 2. Load Components

        await this.loadCommands(pluginPath, namespace);
        await this.loadSkills(pluginPath, namespace);
        await this.loadAgents(pluginPath, namespace);
        await this.loadHooks(pluginPath, namespace);
        await this.loadLspServers(pluginPath, namespace);
    }

    private static async loadCommands(pluginPath: string, namespace: string) {
        const commandsDir = path.join(pluginPath, 'commands');
        if (fs.existsSync(commandsDir)) {
            const files = fs.readdirSync(commandsDir).filter(f => f.endsWith('.md'));
            for (const file of files) {
                const commandName = path.basename(file, '.md');
                this.registerSkillCommand(path.join(commandsDir, file), namespace, commandName);
            }
        }
    }

    private static async loadSkills(pluginPath: string, namespace: string) {
        const skillsDir = path.join(pluginPath, 'skills');
        if (fs.existsSync(skillsDir)) {
            try {
                const items = fs.readdirSync(skillsDir);
                for (const item of items) {
                    const itemPath = path.join(skillsDir, item);
                    if (fs.statSync(itemPath).isDirectory()) {
                        const skillFile = path.join(itemPath, 'SKILL.md');
                        if (fs.existsSync(skillFile)) {
                            this.registerSkillCommand(skillFile, namespace, item);
                        }
                    }
                }
            } catch (e) {
                // Ignore if skills dir is empty or fails
            }
        }
    }

    private static registerSkillCommand(filePath: string, namespace: string, commandName: string) {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const parsed = matter(content);
            const description = parsed.data.description || "";
            const fullCommandName = `${namespace}:${commandName}`;

            commandRegistry.register({
                name: fullCommandName,
                description,
                type: 'prompt',
                userFacingName: () => fullCommandName,
                isEnabled: () => true,
                isHidden: false,
                progressMessage: "Running plugin command...",
                source: "plugin",
                getPromptForCommand: async (args) => {
                    let prompt = parsed.content;
                    if (args) {
                        prompt = prompt.replace(/\$ARGUMENTS/g, args);
                    }
                    return [{ type: 'text', text: prompt }];
                }
            });
        } catch (e) {
            terminalLog(`Failed to register skill ${commandName}: ${e}`, "error");
        }
    }

    private static async loadAgents(pluginPath: string, namespace: string) {
        const agentsDir = path.join(pluginPath, 'agents');
        if (fs.existsSync(agentsDir)) {
            try {
                const files = fs.readdirSync(agentsDir).filter(f => f.endsWith('.md'));
                for (const file of files) {
                    const agent = loadAgent(path.join(agentsDir, file), 'project');
                    if (agent) {
                        // Prefix agent name with namespace? Docs say "agents appear in /agents".
                        // Usually agents are singular. Docs: "Plugin name is namespace... /plug:hello".
                        // For agents, maybe they become available as `--agent plug:reviewer`?
                        // AgentPersistence uses agentType (filename).
                        // I should probably ensure uniqueness or just register as is?
                        // Docs don't specify agent namespacing strictly, but it's good practice.
                        registerAgent({ ...agent, scope: 'plugin' } as any);
                    }
                }
            } catch (e) { }
        }
    }

    private static async loadHooks(pluginPath: string, namespace: string) {
        const hooksPath = path.join(pluginPath, 'hooks', 'hooks.json');
        if (fs.existsSync(hooksPath)) {
            try {
                const hooksConfig = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
                if (hooksConfig.hooks) {
                    for (const [event, hooks] of Object.entries(hooksConfig.hooks)) {
                        if (Array.isArray(hooks)) {
                            for (const hook of hooks) {
                                hookService.registerHook(event as any, hook as any);
                            }
                        }
                    }
                }
            } catch (e) {
                terminalLog(`Failed to load hooks from ${hooksPath}: ${e}`, "error");
            }
        }
    }

    private static async loadLspServers(pluginPath: string, namespace: string) {
        // 1. Check for .lsp.json
        const lspConfigPath = path.join(pluginPath, '.lsp.json');
        const lspConfigs: Record<string, LspServerConfig> = {};

        if (fs.existsSync(lspConfigPath)) {
            try {
                const config = JSON.parse(fs.readFileSync(lspConfigPath, 'utf8'));
                Object.assign(lspConfigs, config);
            } catch (e) {
                terminalLog(`Failed to load LSP config from ${lspConfigPath}: ${e}`, "error");
            }
        }

        // 2. Check manifest for inline configs or file references
        // We assume manifest is already parsed, but here we read it again or pass it down. 
        // For simplicity, we re-read or rely on the previous method having access?
        // loadPluginFromDirectory reads it. But we don't pass it.
        // Let's re-read simply as it is small.
        const manifestPath = path.join(pluginPath, '.claude-plugin', 'plugin.json');
        if (fs.existsSync(manifestPath)) {
            try {
                const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
                if (manifest.lspServers) {
                    // lspServers can be path string, array of paths/objects, or object map
                    const entries = Array.isArray(manifest.lspServers) ? manifest.lspServers : [manifest.lspServers];

                    for (const entry of entries) {
                        if (typeof entry === 'string') {
                            // Path to config file
                            const configPath = path.resolve(pluginPath, entry);
                            if (fs.existsSync(configPath)) {
                                try {
                                    const loaded = JSON.parse(fs.readFileSync(configPath, 'utf8'));
                                    Object.assign(lspConfigs, loaded);
                                } catch (e) {
                                    terminalLog(`Failed to load referenced LSP config ${entry}: ${e}`, "warn");
                                }
                            }
                        } else if (typeof entry === 'object') {
                            // Can be inline map (name -> config) or just config object?
                            // 2.1.19 chunk458 MOA is specific server config.
                            // chunk909: 
                            // if entry is object: iterate keys.
                            for (const [name, config] of Object.entries(entry)) {
                                // Simple validation
                                if ((config as any).command) {
                                    lspConfigs[name] = config as LspServerConfig;
                                    // Ensure name is set
                                    lspConfigs[name].name = name;
                                }
                            }
                        }
                    }
                }
            } catch (e) {
                // Ignore
            }
        }

        // Register all found configs
        const serverManager = LspServerManager.getInstance();
        for (const [name, config] of Object.entries(lspConfigs)) {
            // Apply plugin namespace if not global? 
            // 2.1.19 doesn't seems to namespace LSP servers strictly, 
            // but conflicts might happen.
            // For now, register as is.
            serverManager.registerServerConfig({
                ...config,
                name // Ensure name matches map key
            });
        }
    }

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
            // Read current version
            const packageJsonPath = path.join(installPath, 'package.json');
            let oldVersion = '0.0.0';
            if (fs.existsSync(packageJsonPath)) {
                try {
                    oldVersion = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8')).version || '0.0.0';
                } catch { }
            }

            // Perform pull and capture output
            const { execSync } = await import('child_process');
            const pullOutput = execSync('git pull', {
                cwd: installPath,
                encoding: 'utf8',
                stdio: ['ignore', 'pipe', 'ignore']
            });

            const alreadyUpToDate = pullOutput.includes('Already up to date');
            console.log(`[Plugins] Git pull output: ${pullOutput.trim()}`);

            let newVersion = oldVersion;
            if (fs.existsSync(packageJsonPath)) {
                try {
                    newVersion = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8')).version || '0.0.0';
                } catch { }
            }

            const versionChanged = semver.gt(newVersion, oldVersion);

            // Re-install dependencies if version changed or requested
            if (versionChanged || !alreadyUpToDate) {
                if (fs.existsSync(packageJsonPath)) {
                    console.log(`[Plugins] Installing dependencies for ${pluginId}...`);
                    try {
                        const { execSync } = await import('child_process');
                        // Check if npm is available
                        let npmAvailable = false;
                        try {
                            execSync('npm --version', { stdio: 'ignore' });
                            npmAvailable = true;
                        } catch (npmCheckErr) {
                            console.error(`[Plugins] npm is not available: ${(npmCheckErr as Error).message}`);
                        }

                        if (npmAvailable) {
                            try {
                                // Use stdio: 'pipe' to capture output for logging on failure
                                execSync('npm install --no-audit --no-fund --omit=dev', {
                                    cwd: installPath,
                                    encoding: 'utf8',
                                    stdio: ['ignore', 'pipe', 'pipe']
                                });
                            } catch (installErr: any) {
                                const stderr = installErr.stderr?.toString() || '';
                                const message = `[Plugins] npm install failed for ${pluginId}: ${installErr.message}\n${stderr}`;
                                console.error(message);
                                // We don't throw here to allow partial success, but we log it
                            }
                        }
                    } catch (err) {
                        console.error(`[Plugins] Dependency installation check failed: ${(err as Error).message}`);
                    }
                }
            }

            return {
                success: true,
                message: alreadyUpToDate ? `Plugin ${pluginId} is already up to date` : `Updated plugin ${pluginId} to ${newVersion}`,
                alreadyUpToDate,
                oldVersion,
                newVersion,
                versionChanged
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
                    const { execSync } = await import('child_process');

                    // Check if npm is available
                    let npmAvailable = false;
                    try {
                        execSync('npm --version', { stdio: 'ignore' });
                        npmAvailable = true;
                    } catch (npmCheckErr) {
                        console.error(`[Plugins] npm is not available: ${(npmCheckErr as Error).message}`);
                    }

                    if (npmAvailable) {
                        try {
                            execSync('npm install --no-audit --no-fund --omit=dev', {
                                cwd: installPath,
                                encoding: 'utf8',
                                stdio: ['ignore', 'pipe', 'pipe']
                            });
                        } catch (installErr: any) {
                            const stderr = installErr.stderr?.toString() || '';
                            console.error(`[Plugins] Failed to install dependencies for ${pluginId}: ${installErr.message}\n${stderr}`);
                            // We continue even if npm fails, as the plugin might work without new dependencies 
                            // or user can fix it manually.
                        }
                    }
                } catch (e) {
                    console.error(`[Plugins] Failed to install dependencies: ${(e as Error).message}`);
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
