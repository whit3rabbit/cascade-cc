/**
 * File: src/services/mcp/McpClientManager.ts
 * Role: Central management interface for Model Context Protocol (MCP) clients and plugin state.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { WebSocketClientTransport } from "@modelcontextprotocol/sdk/client/websocket.js";
import { getSettings } from "../config/SettingsService.js";
import { EnvService } from "../config/EnvService.js";
import { McpServerManager, McpServerConfig } from "./McpServerManager.js";
import { MCP_PROXY_PATH, MCP_PROXY_URL } from "../../constants/product.js";
import { getSessionId } from "../../utils/shared/runtimeAndEnv.js";

interface ActiveClient {
    client: Client;
    transport: any;
    config: McpServerConfig;
    capabilities: any;
    cleanup: () => Promise<void>;
    cache: {
        tools?: any[];
        prompts?: any[];
        resources?: any[];
    };
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
        if (this.activeClients.has(serverId)) {
            console.log(`[MCP] Server ${serverId} already connected, disconnecting first...`);
            await this.disconnect(serverId);
        }

        const { expanded: expandedConfig, missingVars } = McpServerManager.expandConfig(config);

        if (missingVars.length > 0) {
            console.warn(`[MCP] Server ${serverId} missing environment variables: ${missingVars.join(", ")}`);
        }

        console.log(`[MCP] Connecting to server: ${serverId} (${expandedConfig.type || 'stdio'})`);

        try {
            let transport: any;

            if (expandedConfig.type === "stdio" || !expandedConfig.type) {
                if (!expandedConfig.command) {
                    throw new Error(`MCP Server ${serverId} missing 'command' for stdio transport`);
                }

                transport = new StdioClientTransport({
                    command: expandedConfig.command,
                    args: expandedConfig.args || [],
                    env: { ...(process.env as Record<string, string>), ...(expandedConfig.env || {}) }
                });
            } else if (expandedConfig.type === "sse") {
                if (!expandedConfig.url) throw new Error(`MCP Server ${serverId} missing 'url' for transport`);
                transport = new SSEClientTransport(new URL(expandedConfig.url));
            } else if (expandedConfig.type === "ws") {
                if (!expandedConfig.url) throw new Error(`MCP Server ${serverId} missing 'url' for ws transport`);
                transport = new WebSocketClientTransport(new URL(expandedConfig.url));
            } else if (expandedConfig.type === "claudeai-proxy" || expandedConfig.type === "http") {
                // Handling for claudeai-proxy and http which might need auth
                let targetUrl = expandedConfig.url;

                if (expandedConfig.type === "claudeai-proxy") {
                    const { OAuthService } = await import('../auth/OAuthService.js');

                    const proxyUrlBase = EnvService.get('MCP_PROXY_URL') || MCP_PROXY_URL;
                    const proxyPath = EnvService.get('MCP_PROXY_PATH') || MCP_PROXY_PATH;
                    const proxyServerId = expandedConfig.id || serverId;
                    if (!proxyServerId) {
                        throw new Error(`MCP Server ${serverId} missing 'id' for claudeai-proxy transport`);
                    }

                    targetUrl = `${proxyUrlBase}${proxyPath.replace('{server_id}', proxyServerId)}`;

                    // Custom transport options with Auth
                    transport = new SSEClientTransport(new URL(targetUrl), {
                        eventSourceInit: {
                            fetch: async (input: any, init: any) => {
                                const accessToken = await OAuthService.getValidToken();
                                if (!accessToken) {
                                    throw new Error("No claude.ai OAuth token found");
                                }

                                const headers = new Headers(init?.headers);
                                headers.set('Authorization', `Bearer ${accessToken}`);
                                headers.set('X-Mcp-Client-Session-Id', getSessionId() || 'unknown');
                                headers.set('User-Agent', `claude-code/${process.env.npm_package_version || 'unknown'}`);

                                return fetch(input, {
                                    ...init,
                                    headers
                                });
                            }
                        }
                    } as any);
                } else {
                    // Standard HTTP/SSE
                    if (!targetUrl) throw new Error(`MCP Server ${serverId} missing 'url'`);
                    transport = new SSEClientTransport(new URL(targetUrl));
                }
            } else {
                throw new Error(`Unsupported transport type: ${expandedConfig.type}`);
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

            const mcpTimeout = Number(EnvService.get('MCP_TIMEOUT')) || 60000;

            // Connect with timeout
            await Promise.race([
                client.connect(transport),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`MCP server connection timed out after ${mcpTimeout}ms`)), mcpTimeout))
            ]);

            const capabilities = client.getServerCapabilities();
            console.log(`[MCP] Connected to ${serverId}. Capabilities:`, capabilities);

            const activeClient: ActiveClient = {
                client,
                transport,
                config: expandedConfig,
                capabilities,
                cleanup: async () => {
                    try {
                        await client.close();
                    } catch {
                        // ignore errors on close
                    }
                },
                cache: {}
            };

            // Register notification handlers for list_changed
            if (capabilities?.tools?.listChanged) {
                client.setNotificationHandler({ method: "notifications/tools/list_changed" } as any, async () => {
                    console.log(`[MCP] Received tools/list_changed from ${serverId}, refreshing...`);
                    activeClient.cache.tools = undefined;
                    // Proactively refresh tools
                    await this.getToolsForClient(serverId);
                });
            }

            if (capabilities?.prompts?.listChanged) {
                client.setNotificationHandler({ method: "notifications/prompts/list_changed" } as any, async () => {
                    console.log(`[MCP] Received prompts/list_changed from ${serverId}, refreshing...`);
                    activeClient.cache.prompts = undefined;
                    await this.getPromptsForClient(serverId);
                });
            }

            if (capabilities?.resources?.listChanged) {
                client.setNotificationHandler({ method: "notifications/resources/list_changed" } as any, async () => {
                    console.log(`[MCP] Received resources/list_changed from ${serverId}, refreshing...`);
                    activeClient.cache.resources = undefined;
                    await this.getResourcesForClient(serverId);
                });
            }

            this.activeClients.set(serverId, activeClient);

        } catch (error) {
            console.error(`[MCP] Failed to connect to ${serverId}:`, error);
            throw error;
        }
    }

    /**
     * Initializes all configured MCP servers.
     */
    async initializeAllServers(): Promise<void> {
        console.log("[MCP] Initializing all servers...");
        const allServers = await McpServerManager.getAllMcpServers();

        for (const [name, config] of Object.entries(allServers)) {
            if (McpServerManager.isMcpServerDenied(name)) {
                console.log(`[MCP] Server ${name} is denied, skipping...`);
                continue;
            }
            if (!McpServerManager.isMcpServerEnabled(name, config)) {
                console.log(`[MCP] Server ${name} is disabled, skipping...`);
                continue;
            }

            try {
                await this.connect(name, config);
            } catch (err) {
                console.error(`[MCP] Auto-connect failed for ${name}:`, err);
            }
        }
    }

    /**
     * Restarts a specific MCP server.
     */
    async restart(serverId: string): Promise<void> {
        const active = this.activeClients.get(serverId);
        const settings = getSettings();
        const config = active?.config || settings.mcp?.servers?.[serverId];

        if (!config) {
            throw new Error(`Configuration for server ${serverId} not found`);
        }

        console.log(`[MCP] Restarting server: ${serverId}`);
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
        for (const serverId of this.activeClients.keys()) {
            const tools = await this.getToolsForClient(serverId);
            allTools.push(...tools);
        }
        return allTools;
    }

    /**
     * Gets tools for a specific client, using cache if available.
     */
    async getToolsForClient(serverId: string): Promise<any[]> {
        const active = this.activeClients.get(serverId);
        if (!active) return [];

        if (active.cache.tools) {
            return active.cache.tools;
        }

        try {
            const toolTimeout = Number(EnvService.get('MCP_TOOL_TIMEOUT')) || 300000;
            const toolsResult = await active.client.listTools();
            const adaptedTools = toolsResult.tools.map(tool => ({
                name: tool.name,
                description: tool.description,
                input_schema: tool.inputSchema,
                serverId,
                isMcp: true, // Mark as MCP tool for deferred search logic
                prompt: async () => tool.description || "", // Simple prompt implementation
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
            active.cache.tools = adaptedTools;
            return adaptedTools;
        } catch (err) {
            console.error(`[MCP] Failed to list tools for ${serverId}:`, err);
            return [];
        }
    }

    /**
     * Aggregates resources from all connected MCP servers.
     */
    async getResources(): Promise<any[]> {
        const allResources: any[] = [];
        for (const serverId of this.activeClients.keys()) {
            const resources = await this.getResourcesForClient(serverId);
            allResources.push(...resources);
        }
        return allResources;
    }

    /**
     * Gets resources for a specific client, using cache if available.
     */
    async getResourcesForClient(serverId: string): Promise<any[]> {
        const active = this.activeClients.get(serverId);
        if (!active) return [];

        if (active.cache.resources) {
            return active.cache.resources;
        }

        try {
            const capabilities = active.client.getServerCapabilities();
            if (capabilities?.resources) {
                const res = await active.client.listResources();
                const resourcesWithId = res.resources.map(r => ({ ...r, serverId }));
                active.cache.resources = resourcesWithId;
                return resourcesWithId;
            }
        } catch (err) {
            console.error(`[MCP] Failed to list resources for ${serverId}:`, err);
        }
        return [];
    }

    /**
     * Aggregates prompts from all connected MCP servers.
     */
    async getPrompts(): Promise<any[]> {
        const allPrompts: any[] = [];
        for (const serverId of this.activeClients.keys()) {
            const prompts = await this.getPromptsForClient(serverId);
            allPrompts.push(...prompts);
        }
        return allPrompts;
    }

    /**
     * Gets prompts for a specific client, using cache if available.
     */
    async getPromptsForClient(serverId: string): Promise<any[]> {
        const active = this.activeClients.get(serverId);
        if (!active) return [];

        if (active.cache.prompts) {
            return active.cache.prompts;
        }

        try {
            const capabilities = active.client.getServerCapabilities();
            if (capabilities?.prompts) {
                const res = await active.client.listPrompts();
                const promptsWithId = res.prompts.map(p => ({ ...p, serverId }));
                active.cache.prompts = promptsWithId;
                return promptsWithId;
            }
        } catch (err) {
            console.error(`[MCP] Failed to list prompts for ${serverId}:`, err);
        }
        return [];
    }

    /**
     * Reads a resource from a specific MCP server.
     */
    async readResource(serverId: string, uri: string): Promise<any> {
        const client = this.getClient(serverId);
        if (!client) throw new Error(`Server ${serverId} not found`);
        return await client.readResource({ uri });
    }

    /**
     * Retrieves a prompt from a specific MCP server.
     */
    async getPrompt(serverId: string, name: string, args?: Record<string, string>): Promise<any> {
        const client = this.getClient(serverId);
        if (!client) throw new Error(`Server ${serverId} not found`);
        return await client.getPrompt({ name, arguments: args });
    }
}

/**
 * Global initialization for the MCP client ecosystem.
 */
export const mcpClientManager = new McpClientManager();

export function initializeClientManager(): McpClientManager {
    return mcpClientManager;
}
