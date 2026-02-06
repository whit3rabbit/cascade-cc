/**
 * File: src/services/mcp/McpServerManager.ts
 * Role: Manages MCP server configurations across multiple scopes (enterprise, local, project, user, dynamic).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { z } from 'zod';
import { getSettings, updateSettings } from '../config/SettingsService.js';
import { getEnterpriseMcpConfigPath } from '../../utils/shared/runtimeAndEnv.js';

// --- MCP Server Configuration Schemas ---

const StdioConfigSchema = z.object({
    type: z.literal("stdio").optional(),
    command: z.string().min(1, "Command cannot be empty"),
    args: z.array(z.string()).default([]),
    env: z.record(z.string(), z.string()).optional()
});

const SseConfigSchema = z.object({
    type: z.literal("sse"),
    url: z.string().url(),
    headers: z.record(z.string(), z.string()).optional()
});

const HttpConfigSchema = z.object({
    type: z.literal("http"),
    url: z.string().url(),
    headers: z.record(z.string(), z.string()).optional()
});

const WsConfigSchema = z.object({
    type: z.literal("ws"),
    url: z.string().url(),
    headers: z.record(z.string(), z.string()).optional()
});

const IdeConfigSchema = z.object({
    type: z.union([z.literal("sse-ide"), z.literal("ws-ide")]),
});

const SdkConfigSchema = z.object({
    type: z.literal("sdk"),
    name: z.string().optional()
});

const ClaudeAiProxyConfigSchema = z.object({
    type: z.literal("claudeai-proxy"),
    url: z.string().url(),
    id: z.string()
});

export const McpServerConfigSchema = z.union([
    StdioConfigSchema,
    SseConfigSchema,
    HttpConfigSchema,
    WsConfigSchema,
    IdeConfigSchema,
    SdkConfigSchema,
    ClaudeAiProxyConfigSchema
]);

export type McpServerScope = "enterprise" | "local" | "project" | "user" | "dynamic" | "claudeai";

export type McpServerConfig = z.infer<typeof McpServerConfigSchema> & {
    scope?: McpServerScope;
    enabled?: boolean;
    id?: string; // For claudeai-proxy
};

/**
 * Orchestrates MCP server configurations and persistent storage.
 */
export class McpServerManager {
    private static dynamicServers: Record<string, McpServerConfig> = {};

    /**
     * Expands environmental variables in a string (${VAR} or ${VAR:-default}).
     */
    static expandVariables(str: string): { expanded: string; missingVars: string[] } {
        const missingVars: string[] = [];
        const expanded = str.replace(/\$\{([^}]+)\}/g, (match, content) => {
            const [varName, defaultValue] = content.split(":-", 2);
            const envValue = process.env[varName];
            if (envValue !== undefined) return envValue;
            if (defaultValue !== undefined) return defaultValue;
            missingVars.push(varName);
            return match;
        });
        return { expanded, missingVars };
    }

    /**
     * Deeply expands variables in an MCP server configuration.
     */
    static expandConfig(config: McpServerConfig): { expanded: McpServerConfig; missingVars: string[] } {
        const missingVars: string[] = [];
        const expand = (str: string) => {
            const { expanded, missingVars: vars } = this.expandVariables(str);
            missingVars.push(...vars);
            return expanded;
        };

        const expanded: any = { ...config };

        if ('command' in expanded && expanded.command) expanded.command = expand(expanded.command);
        if ('args' in expanded && expanded.args) expanded.args = expanded.args.map((a: string) => expand(a));
        if ('url' in expanded && expanded.url) expanded.url = expand(expanded.url);

        if ('env' in expanded && expanded.env) {
            const newEnv: Record<string, string> = {};
            for (const [k, v] of Object.entries(expanded.env as Record<string, string>)) {
                newEnv[k] = expand(v);
            }
            expanded.env = newEnv;
        }

        if ('headers' in expanded && expanded.headers) {
            const newHeaders: Record<string, string> = {};
            for (const [k, v] of Object.entries(expanded.headers as Record<string, string>)) {
                newHeaders[k] = expand(v);
            }
            expanded.headers = newHeaders;
        }

        return { expanded: expanded as McpServerConfig, missingVars: [...new Set(missingVars)] };
    }

    /**
     * Checks if a server is explicitly denied by user settings.
     */
    static isMcpServerDenied(serverName: string): boolean {
        const settings = getSettings();
        const denied = settings.mcp?.deniedServers || [];
        return denied.includes(serverName);
    }

    /**
     * Checks if a server is enabled in its configuration.
     */
    static isMcpServerEnabled(serverName: string, config?: McpServerConfig): boolean {
        if (config && config.enabled === false) return false;
        const settings = getSettings();
        const serverSettings = settings.mcp?.servers?.[serverName] as any;
        if (serverSettings && serverSettings.enabled === false) return false;
        return true;
    }

    /**
     * Adds or updates an MCP server configuration in a specific scope.
     */
    static async addMcpServer(name: string, config: any, scope: McpServerScope) {
        if (name.match(/[^a-zA-Z0-9_: @-]/)) {
            throw new Error(`Invalid MCP server name: ${name}`);
        }

        const validConfig = McpServerConfigSchema.parse(config);

        if (scope === "dynamic") {
            this.dynamicServers[name] = validConfig;
            return;
        }

        // Persistent scopes
        const settings = getSettings();
        if (!settings.mcp) settings.mcp = { servers: {} };
        if (!settings.mcp.servers) settings.mcp.servers = {};

        settings.mcp.servers[name] = { ...validConfig, scope };
        updateSettings(settings);
    }

    /**
     * Loads servers from the enterprise configuration if it exists.
     */
    private static async loadEnterpriseServers(): Promise<Record<string, McpServerConfig>> {
        const configPath = getEnterpriseMcpConfigPath();
        if (!fs.existsSync(configPath)) return {};
        try {
            const content = fs.readFileSync(configPath, 'utf8');
            const data = JSON.parse(content);
            const servers: Record<string, McpServerConfig> = {};
            if (data.mcpServers) {
                for (const [name, config] of Object.entries(data.mcpServers)) {
                    const parsed = McpServerConfigSchema.safeParse(config);
                    if (parsed.success) {
                        servers[name] = { ...parsed.data, scope: "enterprise" };
                    }
                }
            }
            return servers;
        } catch (e) {
            console.error(`Failed to load enterprise MCP config: ${e}`);
            return {};
        }
    }

    /**
     * Loads servers from the project configuration (.mcp.json).
     * Searches upwards from the current working directory.
     */
    private static async loadProjectServers(): Promise<Record<string, McpServerConfig>> {
        let currentDir = process.cwd();
        const servers: Record<string, McpServerConfig> = {};

        while (true) {
            const configPath = path.join(currentDir, '.mcp.json');
            if (fs.existsSync(configPath)) {
                try {
                    const content = fs.readFileSync(configPath, 'utf8');
                    const data = JSON.parse(content);
                    if (data.mcpServers) {
                        for (const [name, config] of Object.entries(data.mcpServers)) {
                            const parsed = McpServerConfigSchema.safeParse(config);
                            if (parsed.success) {
                                // Only add if not already added (deeper .mcp.json overrides higher ones)
                                if (!servers[name]) {
                                    servers[name] = { ...parsed.data, scope: "project" };
                                }
                            }
                        }
                    }
                } catch (e) {
                    console.error(`Failed to load project MCP config at ${configPath}: ${e}`);
                }
            }
            const parentDir = path.dirname(currentDir);
            if (parentDir === currentDir) break;
            currentDir = parentDir;
        }
        return servers;
    }

    /**
     * Retrieves all MCP servers from all scopes with correct priority.
     * Priority (Highest to Lowest): Enterprise > Local > Project > User > Dynamic
     */
    static async getAllMcpServers(): Promise<Record<string, McpServerConfig>> {
        const enterpriseServers = await this.loadEnterpriseServers();
        const projectServers = await this.loadProjectServers();

        const settings = getSettings();
        const persistentServers = settings.mcp?.servers || {};

        // Local servers (those added via 'claude mcp add' with local scope)
        // User servers (those added via 'claude mcp add' with user scope)
        const localServers: Record<string, McpServerConfig> = {};
        const userServers: Record<string, McpServerConfig> = {};

        for (const [name, config] of Object.entries(persistentServers)) {
            const mcpConfig = config as McpServerConfig;
            if (mcpConfig.scope === "local") localServers[name] = mcpConfig;
            else if (mcpConfig.scope === "user") userServers[name] = mcpConfig;
        }

        // Merge in order of priority (lower priority values are overwritten by higher ones)
        return {
            ...userServers,
            ...projectServers,
            ...localServers,
            ...this.dynamicServers,
            ...enterpriseServers
        };
    }
}
