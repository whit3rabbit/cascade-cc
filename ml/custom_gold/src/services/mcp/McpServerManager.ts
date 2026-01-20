import * as fs from 'node:fs';
import * as path from 'node:path';
import { z } from 'zod';
import { getSettings, updateSettings, SettingsSource, getSettingsPath } from '../terminal/settings.js';
import { log } from '../logger/loggerService.js';
import { getInstalledPlugins } from './McpClientManager.js';
import { expandEnvVars } from './McpBundleLoader.js';

const logger = log("mcp-server-manager");

// --- Schemas from chunk_338.ts (ba and related) ---

const StdioConfigSchema = z.object({
    type: z.literal("stdio").optional(),
    command: z.string().min(1, "Command cannot be empty"),
    args: z.array(z.string()).default([]),
    env: z.record(z.string(), z.string()).optional()
});

const SseConfigSchema = z.object({
    type: z.literal("sse"),
    url: z.string().url(),
    headers: z.record(z.string(), z.string()).optional(),
    headersHelper: z.string().optional()
});

const SseIdeConfigSchema = z.object({
    type: z.literal("sse-ide"),
    url: z.string().url(),
    ideName: z.string(),
    ideRunningInWindows: z.boolean().optional()
});

const WsIdeConfigSchema = z.object({
    type: z.literal("ws-ide"),
    url: z.string(), // Might not be full URL schema due to ws://
    ideName: z.string(),
    authToken: z.string().optional(),
    ideRunningInWindows: z.boolean().optional()
});

const HttpConfigSchema = z.object({
    type: z.literal("http"),
    url: z.string().url(),
    headers: z.record(z.string(), z.string()).optional(),
    headersHelper: z.string().optional()
});

const WsConfigSchema = z.object({
    type: z.literal("ws"),
    url: z.string(),
    headers: z.record(z.string(), z.string()).optional(),
    headersHelper: z.string().optional()
});

const SdkConfigSchema = z.object({
    type: z.literal("sdk"),
    name: z.string()
});

const ClaudeAiProxyConfigSchema = z.object({
    type: z.literal("claudeai-proxy"),
    url: z.string().url(),
    id: z.string()
});

export const McpServerConfigSchema = z.union([
    StdioConfigSchema,
    SseConfigSchema,
    SseIdeConfigSchema,
    WsIdeConfigSchema,
    HttpConfigSchema,
    WsConfigSchema,
    SdkConfigSchema,
    ClaudeAiProxyConfigSchema
]);

export type McpServerConfig = z.infer<typeof McpServerConfigSchema> & {
    scope?: "project" | "user" | "local" | "dynamic" | "enterprise" | "claudeai";
};

const McpConfigRootSchema = z.object({
    mcpServers: z.record(z.string(), McpServerConfigSchema)
});

// --- Implementation ---

export class McpServerManager {
    static dynamicServers: Record<string, McpServerConfig> = {};

    static isMcpServerDenied(serverName: string, _config: any): boolean {
        const settings = getSettings("userSettings");
        if (!settings.deniedMcpServers) return false;
        return settings.deniedMcpServers.includes(serverName);
    }

    static isMcpServerAllowed(serverName: string, config: any): boolean {
        if (this.isMcpServerDenied(serverName, config)) return false;
        const settings = getSettings("userSettings");
        if (!settings.allowedMcpServers) return true;
        if (settings.allowedMcpServers.length === 0) return false;
        return settings.allowedMcpServers.includes(serverName);
    }

    static async addMcpServer(name: string, config: any, scope: string) {
        if (name.match(/[^a-zA-Z0-9_-]/)) {
            throw new Error(`Invalid MCP server name: ${name}. Names must only contain alphanumeric characters, underscores, and hyphens.`);
        }
        if (this.isMcpServerDenied(name, config)) {
            throw new Error(`Adding MCP server "${name}" is blocked by security policy.`);
        }

        const validConfig = McpServerConfigSchema.parse(config);

        switch (scope) {
            case "project":
                await this.updateProjectMcpConfig(name, validConfig);
                break;
            case "user":
                await this.updateUserSettingsMcpConfig(name, validConfig);
                break;
            case "local":
                await this.updateLocalSettingsMcpConfig(name, validConfig);
                break;
            case "dynamic":
                this.dynamicServers[name] = validConfig;
                break;
            default:
                throw new Error(`Cannot add MCP server to unsupported scope: ${scope}`);
        }
    }

    static async removeMcpServer(name: string, scope: string) {
        switch (scope) {
            case "project":
                await this.removeFromProjectMcpConfig(name);
                break;
            case "user":
                await this.removeFromUserSettingsMcpConfig(name);
                break;
            case "local":
                await this.removeFromLocalSettingsMcpConfig(name);
                break;
            case "dynamic":
                delete this.dynamicServers[name];
                break;
            default:
                throw new Error(`Cannot remove MCP server from unsupported scope: ${scope}`);
        }
    }

    static async getAllMcpServers(): Promise<{ servers: Record<string, McpServerConfig>; errors: any[] }> {
        const enterprise = this.loadServersFromScope("enterprise");
        const user = this.loadServersFromScope("user");
        const project = this.loadServersFromScope("project");
        const local = this.loadServersFromScope("local");

        const pluginServers: Record<string, McpServerConfig> = {};
        const pluginErrors: any[] = [];

        try {
            const plugins = await getInstalledPlugins();
            for (const plugin of plugins.enabled) {
                const config = await this.loadMcpConfigFromPlugin(plugin);
                if (config) {
                    Object.assign(pluginServers, config);
                }
            }
        } catch (err) {
            logger.error(`Error loading plugin MCP servers: ${err}`);
        }

        const aggregated: Record<string, McpServerConfig> = {
            ...this.dynamicServers,
            ...enterprise.servers,
            ...pluginServers,
            ...user.servers,
            ...project.servers,
            ...local.servers
        };

        const filtered: Record<string, McpServerConfig> = {};
        for (const [name, config] of Object.entries(aggregated)) {
            if (this.isMcpServerAllowed(name, config)) {
                filtered[name] = config;
            }
        }

        const allErrors = [
            ...enterprise.errors,
            ...user.errors,
            ...project.errors,
            ...local.errors,
            ...pluginErrors
        ];

        return { servers: filtered, errors: allErrors };
    }

    private static loadServersFromScope(scope: string): { servers: Record<string, McpServerConfig>; errors: any[] } {
        let rawServers: Record<string, any> = {};
        let errors: any[] = [];

        if (scope === "project") {
            const projectRoot = process.cwd();
            const mcpPath = path.join(projectRoot, ".mcp.json");
            if (fs.existsSync(mcpPath)) {
                try {
                    const content = fs.readFileSync(mcpPath, "utf-8");
                    const parsed = JSON.parse(content);
                    const result = McpConfigRootSchema.safeParse(parsed);
                    if (result.success) {
                        rawServers = result.data.mcpServers;
                    } else {
                        errors.push({ file: mcpPath, message: "Invalid .mcp.json schema", details: result.error });
                    }
                } catch (err) {
                    errors.push({ file: mcpPath, message: `Failed to read .mcp.json: ${err}` });
                }
            }
        } else if (scope === "user" || scope === "local") {
            const settings = getSettings(scope === "user" ? "userSettings" : "localSettings");
            if (settings.mcpServers) {
                rawServers = settings.mcpServers;
            }
        } else if (scope === "enterprise") {
            // Enterprise policy logic placeholder
        }

        const result: Record<string, McpServerConfig> = {};
        for (const [name, config] of Object.entries(rawServers)) {
            result[name] = { ...config, scope } as McpServerConfig;
            if (config.type === "stdio" || !config.type) {
                try {
                    const { expanded } = expandEnvVars(JSON.stringify(config));
                    result[name] = { ...JSON.parse(expanded), scope } as McpServerConfig;
                } catch {
                    // Fallback to non-expanded if it's not valid JSON yet
                }
            }
        }

        return { servers: result, errors };
    }

    private static async loadMcpConfigFromPlugin(plugin: any): Promise<Record<string, McpServerConfig> | null> {
        if (!plugin.mcpServers) return null;

        const result: Record<string, McpServerConfig> = {};
        for (const [name, config] of Object.entries(plugin.mcpServers)) {
            const pluginInjectedName = `plugin:${plugin.name}:${name}`;
            result[pluginInjectedName] = { ...(config as any), scope: "dynamic" };
        }
        return result;
    }

    private static async updateProjectMcpConfig(name: string, config: McpServerConfig) {
        const mcpPath = path.join(process.cwd(), ".mcp.json");
        let root: any = { mcpServers: {} };
        if (fs.existsSync(mcpPath)) {
            try {
                root = JSON.parse(fs.readFileSync(mcpPath, "utf-8"));
            } catch { }
        }
        root.mcpServers = root.mcpServers || {};
        root.mcpServers[name] = config;
        fs.writeFileSync(mcpPath, JSON.stringify(root, null, 2) + "\n");
    }

    private static async removeFromProjectMcpConfig(name: string) {
        const mcpPath = path.join(process.cwd(), ".mcp.json");
        if (!fs.existsSync(mcpPath)) return;
        try {
            const root = JSON.parse(fs.readFileSync(mcpPath, "utf-8"));
            if (root.mcpServers && root.mcpServers[name]) {
                delete root.mcpServers[name];
                fs.writeFileSync(mcpPath, JSON.stringify(root, null, 2) + "\n");
            }
        } catch { }
    }

    private static async updateUserSettingsMcpConfig(name: string, config: McpServerConfig) {
        const settings = getSettings("userSettings");
        const mcpServers = { ...(settings.mcpServers || {}), [name]: config };
        updateSettings("userSettings", { mcpServers });
    }

    private static async removeFromUserSettingsMcpConfig(name: string) {
        const settings = getSettings("userSettings");
        if (settings.mcpServers && settings.mcpServers[name]) {
            const mcpServers = { ...settings.mcpServers };
            delete mcpServers[name];
            updateSettings("userSettings", { mcpServers });
        }
    }

    private static async updateLocalSettingsMcpConfig(name: string, config: McpServerConfig) {
        const settings = getSettings("localSettings");
        const mcpServers = { ...(settings.mcpServers || {}), [name]: config };
        updateSettings("localSettings", { mcpServers });
    }

    private static async removeFromLocalSettingsMcpConfig(name: string) {
        const settings = getSettings("localSettings");
        if (settings.mcpServers && settings.mcpServers[name]) {
            const mcpServers = { ...settings.mcpServers };
            delete mcpServers[name];
            updateSettings("localSettings", { mcpServers });
        }
    }
}
