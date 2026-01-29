/**
 * File: src/utils/style.ts
 * Role: Utilities for managing MCP endpoint configuration and server status validation.
 */

import { join } from 'node:path';
import { readFileSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { getConfigDir, getSessionId } from './shared/runtimeAndEnv.js';

let currentConfig: any = null;

/**
 * Returns the path to the MCP endpoint configuration file.
 */
function getEndpointConfigPath(): string {
    const sessionId = getSessionId();
    return join(getConfigDir(), `${sessionId}.endpoint`);
}

/**
 * Saves the MCP server endpoint configuration to disk.
 * 
 * @param {any} config - The configuration object to save.
 */
export function saveEndpointConfig(config: any): void {
    if (config) {
        currentConfig = config;
    }

    if (!currentConfig) {
        return;
    }

    const configDir = getConfigDir();
    if (!existsSync(configDir)) {
        mkdirSync(configDir, { recursive: true });
    }

    const filePath = getEndpointConfigPath();
    const encodedConfig = Buffer.from(JSON.stringify(currentConfig)).toString('base64');

    // Save with owner-only permissions (0o600)
    writeFileSync(filePath, encodedConfig, { mode: 0o600 });
}

/**
 * Reads the MCP server endpoint configuration from disk.
 * 
 * @returns {any | null} The decoded configuration or null if not found.
 */
export function readEndpointConfig(): any | null {
    const filePath = getEndpointConfigPath();

    if (!existsSync(filePath)) {
        return null;
    }

    try {
        const fileContent = readFileSync(filePath, 'utf-8');
        return JSON.parse(Buffer.from(fileContent, 'base64').toString('utf-8'));
    } catch (error) {
        return null;
    }
}

/**
 * Finds an element in an array by name, supporting aliases.
 * 
 * @param {any[]} elements - Array of objects with a 'name' property.
 * @param {string} targetName - The name to search for.
 * @param {Record<string, string>} [aliasMap] - Optional map of aliases.
 * @returns {any | undefined}
 */
export function findElementByName(elements: any[], targetName: string, aliasMap?: Record<string, string>): any | undefined {
    // Exact match
    const element = elements.find(el => el.name === targetName);
    if (element) return element;

    // Alias match
    if (aliasMap && aliasMap[targetName]) {
        const realName = aliasMap[targetName];
        return elements.find(el => el.name === realName);
    }

    return undefined;
}

/**
 * Validates the connection status of an MCP server and returns an error if not connected.
 * 
 * @param {string} serverName - The name of the server.
 * @param {string} connectionStatus - The status string.
 * @returns {Error | null}
 */
export function validateServerConnection(serverName: string, connectionStatus: string): Error | null {
    if (!connectionStatus) {
        return new Error(`Server '${serverName}' not found`);
    }

    if (connectionStatus !== 'connected') {
        const errorMessage = connectionStatus === 'needs-auth'
            ? `Server '${serverName}' needs authentication. Run '/mcp' to manage server connections.`
            : `Server '${serverName}' is not connected (${connectionStatus}). Run '/mcp' to manage server connections.`;
        return new Error(errorMessage);
    }

    return null;
}

/**
 * Global context for MCP CLI main thread interaction.
 */
export const globalContext: { mcpCliMain?: () => any } = {
    mcpCliMain: () => null
};

/**
 * Initialization utility (placeholder for future logic).
 */
export function initialize(): void {
    // No-op for now
}
