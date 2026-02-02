/**
 * File: src/services/mcp/McpServerManager.js
 * Role: Manages MCP server configurations across multiple scopes (project, user, local).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { z } from 'zod';
import { getSettings, updateSettings } from '../config/SettingsService.js';



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

export const McpServerConfigSchema = z.union([
    StdioConfigSchema,
    SseConfigSchema,
    HttpConfigSchema
]);

export type McpServerConfig = z.infer<typeof McpServerConfigSchema> & {
    scope?: "project" | "user" | "local" | "dynamic";
};

/**
 * Orchestrates MCP server configurations and persistent storage.
 */
export class McpServerManager {
    private static dynamicServers: Record<string, McpServerConfig> = {};

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
    static isMcpServerEnabled(serverName: string): boolean {
        const settings = getSettings();
        const config = settings.mcp?.servers?.[serverName] as any;
        if (!config) return true; // Default to true for dynamic servers or if missing config
        return config.enabled !== false;
    }

    /**
     * Adds or updates an MCP server configuration in a specific scope.
     */
    static async addMcpServer(name: string, config: any, scope: string) {
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
     * Retrieves all MCP servers from all scopes.
     */
    static async getAllMcpServers() {
        const settings = getSettings();
        const persistentServers = settings.mcp?.servers || {};

        return {
            ...this.dynamicServers,
            ...persistentServers
        };
    }
}