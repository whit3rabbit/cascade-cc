/**
 * File: src/services/mcp/McpClientManager.ts
 * Role: Central management interface for Model Context Protocol (MCP) clients and plugin state.
 */

// Re-exports for convenience (Legacy/Compatibility)
export { MCPServerMultiselectDialog } from '../../components/mcp/MCPServerDialog.js';
export { togglePlugin, enablePlugin, disablePlugin, updatePlugin } from './PluginManager.js';

/**
 * Manages active MCP client instances and their lifecycles.
 */
export class McpClientManager {
    private activeClients: Map<string, any> = new Map();

    constructor() {
        console.log("[MCP] Client Manager initialized");
    }

    /**
     * Connects to a specific MCP server.
     * 
     * @param serverId - The ID of the server to connect to.
     */
    async connect(serverId: string): Promise<void> {
        console.log(`[MCP] Connecting to server: ${serverId}`);
        // Implementation would use McpServerManager to spawn/connect
    }

    /**
     * Disconnects from an active MCP server.
     * 
     * @param serverId - The ID of the server to disconnect.
     */
    async disconnect(serverId: string): Promise<void> {
        console.log(`[MCP] Disconnecting from server: ${serverId}`);
        this.activeClients.delete(serverId);
    }

    /**
     * Lists all currently active and connected MCP clients.
     */
    getActiveClients() {
        return Array.from(this.activeClients.keys());
    }
}

/**
 * Global initialization for the MCP client ecosystem.
 */
export function initializeClientManager(): McpClientManager {
    const manager = new McpClientManager();
    return manager;
}