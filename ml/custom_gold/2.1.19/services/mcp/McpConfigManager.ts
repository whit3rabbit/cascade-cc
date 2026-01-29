/**
 * File: src/services/mcp/McpConfigManager.ts
 * Role: Manages MCP (Model Context Protocol) configuration files in a temporary directory.
 */

import { mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const MCP_TEMP_DIR = process.env.USE_MCP_CLI_DIR || join(tmpdir(), "claude-code-mcp-cli");

export interface McpServerConfig {
    command: string;
    args?: string[];
    env?: Record<string, string>;
    [key: string]: any;
}

export interface McpConfigData {
    mcpServers: Record<string, McpServerConfig>;
}

/**
 * Manages storage and cleanup of MCP client configurations.
 */
export const McpConfigManager = {
    /**
     * Returns the path to the temporary MCP config directory.
     * 
     * @returns {string} The path to the temp directory.
     */
    getTempDir(): string {
        return MCP_TEMP_DIR;
    },

    /**
     * Cleans up the temporary directory.
     * 
     * @returns {Promise<void>}
     */
    async clearTempDir(): Promise<void> {
        try {
            await rm(MCP_TEMP_DIR, { recursive: true, force: true });
        } catch (err) {
            // Ignore cleanup errors
        }
    },

    /**
     * Saves the current MCP configuration (clients and tools) to a JSON file.
     * 
     * @param {string} sessionId - The current session identifier.
     * @param {McpConfigData} configData - The configuration object to persist.
     * @returns {Promise<string | null>} The path to the saved file or null on failure.
     */
    async saveConfiguration(sessionId: string, configData: McpConfigData): Promise<string | null> {
        try {
            await mkdir(MCP_TEMP_DIR, { recursive: true });
            const filePath = join(MCP_TEMP_DIR, `${sessionId}.json`);
            await writeFile(filePath, JSON.stringify(configData, null, 2));
            return filePath;
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            console.error(`[McpConfig] Failed to save config: ${message}`);
            return null;
        }
    }
};
