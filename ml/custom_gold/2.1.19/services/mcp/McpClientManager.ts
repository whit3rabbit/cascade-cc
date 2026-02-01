/**
 * File: src/services/mcp/McpClientManager.ts
 * Role: Central management interface for Model Context Protocol (MCP) clients and plugin state.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { WebSocketClientTransport } from "@modelcontextprotocol/sdk/client/websocket.js";
import { LogManager } from "../logging/LogManager.js";
import { EnvService } from "../config/EnvService.js";

// Re-exports for convenience (Legacy/Compatibility)
export { MCPServerMultiselectDialog } from '../../components/mcp/MCPServerDialog.js';
export { togglePlugin, enablePlugin, disablePlugin, updatePlugin } from './PluginManager.js';

interface McpServerConfig {
    type?: "stdio" | "sse" | "ws" | "http" | "sse-ide" | "ws-ide" | "claudeai-proxy";
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    url?: string;
    authToken?: string;
}

interface ActiveClient {
    client: Client;
    transport: any;
    config: McpServerConfig;
    capabilities: any;
    cleanup: () => Promise<void>;
}

/**
 * Manages active MCP client instances and their lifecycles.
 */
export class McpClientManager {
    private activeClients: Map<string, ActiveClient> = new Map();

    constructor() {
        console.log("[MCP] Client Manager initialized");
    }

    /**
     * Connects to a specific MCP server.
     */
    async connect(serverId: string, config: McpServerConfig): Promise<void> {
        console.log(`[MCP] Connecting to server: ${serverId} (${config.type || 'stdio'})`);

        try {
            let transport: any;

            if (config.type === "stdio" || !config.type) {
                if (!config.command) {
                    throw new Error(`MCP Server ${serverId} missing 'command' for stdio transport`);
                }

                transport = new StdioClientTransport({
                    command: config.command,
                    args: config.args || [],
                    env: { ...(process.env as Record<string, string>), ...(config.env || {}) } // process.env expansion is okay here
                });
            } else if (config.type === "sse") {
                if (!config.url) throw new Error(`MCP Server ${serverId} missing 'url' for sse transport`);
                transport = new SSEClientTransport(new URL(config.url));
            } else if (config.type === "ws") {
                if (!config.url) throw new Error(`MCP Server ${serverId} missing 'url' for ws transport`);
                transport = new WebSocketClientTransport(new URL(config.url));
            } else {
                throw new Error(`Unsupported transport type: ${config.type}`);
            }

            const client = new Client(
                {
                    name: "claude-code",
                    version: "2.1.19"
                },
                {
                    capabilities: {
                        roots: { listChanged: true },
                        sampling: {}
                    }
                }
            );

            const mcpTimeout = Number(EnvService.get('MCP_TIMEOUT')) || 60000; // 60s default
            const connectPromise = client.connect(transport);

            // Timeout wrapper for connect
            await new Promise<void>((resolve, reject) => {
                const timer = setTimeout(() => {
                    reject(new Error(`MCP server connection timed out after ${mcpTimeout}ms`));
                }, mcpTimeout);

                connectPromise.then(() => {
                    clearTimeout(timer);
                    resolve();
                }).catch((err) => {
                    clearTimeout(timer);
                    reject(err);
                });
            });

            const capabilities = client.getServerCapabilities();
            console.log(`[MCP] Connected to ${serverId}. Capabilities:`, capabilities);

            // Add error listeners
            if (config.type === "stdio" || !config.type) {
                // Stdio transport doesn't expose the child process directly here easily
                // but we can listen for transport closure
            }

            this.activeClients.set(serverId, {
                client,
                transport,
                config,
                capabilities,
                cleanup: async () => {
                    try {
                        await client.close();
                    } catch (e) {
                        // ignore errors on close
                    }
                }
            });

        } catch (error) {
            console.error(`[MCP] Failed to connect to ${serverId}:`, error);
            throw error;
        }
    }

    /**
     * Restarts a specific MCP server.
     */
    async restart(serverId: string): Promise<void> {
        const active = this.activeClients.get(serverId);
        if (!active) {
            throw new Error(`Server ${serverId} not found`);
        }

        console.log(`[MCP] Restarting server: ${serverId}`);
        const config = active.config;
        await this.disconnect(serverId);
        await this.connect(serverId, config);
    }

    /**
     * Disconnects from an active MCP server.
     */
    async disconnect(serverId: string): Promise<void> {
        const active = this.activeClients.get(serverId);
        if (active) {
            console.log(`[MCP] Disconnecting from server: ${serverId}`);
            try {
                await active.cleanup();
            } catch (err) {
                console.error(`[MCP] Error disconnecting ${serverId}:`, err);
            }
            this.activeClients.delete(serverId);
        }
    }

    /**
     * Lists all currently active and connected MCP clients.
     */
    getActiveClients(): string[] {
        return Array.from(this.activeClients.keys());
    }

    /**
     * Retrieves the client instance for a server.
     */
    getClient(serverId: string): Client | undefined {
        return this.activeClients.get(serverId)?.client;
    }

    /**
     * Aggregates tools from all connected MCP servers and transforms them
     * into the format expected by ConversationService.
     */
    async getTools(): Promise<any[]> {
        const allTools: any[] = [];
        const toolTimeout = Number(EnvService.get('MCP_TOOL_TIMEOUT')) || 300000; // 5 min default

        for (const [serverId, active] of this.activeClients.entries()) {
            try {
                const toolsResult = await active.client.listTools();
                const adaptedTools = toolsResult.tools.map(tool => ({
                    name: tool.name,
                    description: tool.description,
                    input_schema: tool.inputSchema,
                    serverId,
                    call: async (input: any) => {
                        const callPromise = active.client.callTool({
                            name: tool.name,
                            arguments: input
                        });

                        const result = await new Promise<any>((resolve, reject) => {
                            const timer = setTimeout(() => {
                                reject(new Error(`MCP tool execution timed out after ${toolTimeout}ms`));
                            }, toolTimeout);

                            callPromise.then((res) => {
                                clearTimeout(timer);
                                resolve(res);
                            }).catch((err) => {
                                clearTimeout(timer);
                                reject(err);
                            });
                        });

                        if (result.content && Array.isArray(result.content)) {
                            return result.content.map((c: any) => {
                                if (c.type === 'text') return c.text;
                                if (c.type === 'image') return `[Image: ${c.mimeType}]`;
                                return JSON.stringify(c);
                            }).join('\n');
                        }
                        return JSON.stringify(result);
                    }
                }));
                allTools.push(...adaptedTools);
            } catch (err) {
                console.error(`[MCP] Failed to list tools for ${serverId}:`, err);
            }
        }
        return allTools;
    }

    /**
     * Aggregates resources from all connected MCP servers.
     */
    async getResources(): Promise<any[]> {
        const allResources: any[] = [];
        for (const [serverId, active] of this.activeClients.entries()) {
            try {
                const capabilities = active.client.getServerCapabilities();
                if (capabilities?.resources) {
                    const res = await active.client.listResources();
                    allResources.push(...res.resources.map(r => ({ ...r, serverId })));
                }
            } catch (err) {
                console.error(`[MCP] Failed to list resources for ${serverId}:`, err);
            }
        }
        return allResources;
    }

    /**
     * Aggregates prompts from all connected MCP servers.
     */
    async getPrompts(): Promise<any[]> {
        const allPrompts: any[] = [];
        for (const [serverId, active] of this.activeClients.entries()) {
            try {
                const capabilities = active.client.getServerCapabilities();
                if (capabilities?.prompts) {
                    const res = await active.client.listPrompts();
                    allPrompts.push(...res.prompts.map(p => ({ ...p, serverId })));
                }
            } catch (err) {
                console.error(`[MCP] Failed to list prompts for ${serverId}:`, err);
            }
        }
        return allPrompts;
    }
}

/**
 * Global initialization for the MCP client ecosystem.
 */
/**
 * Global initialization for the MCP client ecosystem.
 */
export const mcpClientManager = new McpClientManager();

export function initializeClientManager(): McpClientManager {
    return mcpClientManager;
}