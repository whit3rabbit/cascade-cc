/**
 * File: src/services/mcp/McpConnectionProvider.tsx
 * Role: Context provider and hooks for managing MCP server connections and state.
 */

import React, { createContext, useContext, ReactNode } from 'react';

interface McpContextValue {
    reconnectMcpServer: (serverId: string) => void;
    toggleMcpServer: (serverId: string, enabled: boolean) => void;
}

const McpContext = createContext<McpContextValue | null>(null);

/**
 * Hook to trigger a reconnection of a specific MCP server.
 */
export function useMcpReconnect() {
    const context = useContext(McpContext);
    if (!context) {
        console.warn("useMcpReconnect must be used within McpConnectionProvider");
        return () => { };
    }
    return context.reconnectMcpServer;
}

/**
 * Hook to toggle the enabled state of a specific MCP server.
 */
export function useMcpToggleEnabled() {
    const context = useContext(McpContext);
    if (!context) {
        console.warn("useMcpToggleEnabled must be used within McpConnectionProvider");
        return () => { };
    }
    return context.toggleMcpServer;
}

interface McpConnectionProviderProps {
    children: ReactNode;
    dynamicMcpConfig?: any;
    isStrictMcpConfig?: boolean;
    mcpCliEndpoint?: string;
}

/**
 * Provider that manages Model Context Protocol (MCP) server connections.
 */
export function McpConnectionProvider({
    children,
    // dynamicMcpConfig, 
    // isStrictMcpConfig, 
    mcpCliEndpoint
}: McpConnectionProviderProps) {

    /**
     * Reconnects to a specific MCP server and updates its status.
     */
    const reconnectMcpServer = (serverId: string) => {
        console.log(`[MCP] Reconnecting to server: ${serverId} (Endpoint: ${mcpCliEndpoint})`);
        // Implementation would interface with McpServerManager
    };

    /**
     * Toggles a server's enabled/disabled state.
     */
    const toggleMcpServer = (serverId: string, enabled: boolean) => {
        console.log(`[MCP] Toggling server ${serverId}: ${enabled ? 'Enabled' : 'Disabled'}`);
    };

    const value = { reconnectMcpServer, toggleMcpServer };

    return (
        <McpContext.Provider value={value}>
            {children}
        </McpContext.Provider>
    );
}

/**
 * Utility to map a transport protocol to a display name.
 */
export function getConnectionType(transportType: string): string {
    switch (transportType.toLowerCase()) {
        case "http": return "HTTP";
        case "ws":
        case "ws-ide": return "WebSocket";
        case "sse": return "SSE";
        default: return transportType.toUpperCase();
    }
}

/**
 * Formats the result of a reconnection attempt into a user-friendly message.
 */
export function getReconnectResultMessage(result: { type: string } | null, connectionType: string) {
    if (!result) return { message: `Failed to reconnect to ${connectionType}.`, success: false };

    switch (result.type) {
        case "connected":
            return { message: `Successfully reconnected to ${connectionType}.`, success: true };
        case "needs-auth":
            return { message: `${connectionType} requires authentication. Use /mcp to authenticate.`, success: false };
        default:
            return { message: `Failed to reconnect to ${connectionType}.`, success: false };
    }
}
