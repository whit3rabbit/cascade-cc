/**
 * File: src/services/mcp/McpBundleLoader.ts
 * Role: Handles loading and environment expansion for MCP server bundles.
 */

import { readFileSync, existsSync } from 'node:fs';
import { EnvService } from '../config/EnvService.js';

/**
 * Expands environment variables in a string (e.g., "${HOME}/path").
 * Aligned with logic found in chunk111/chunk119 config handling.
 * 
 * @param {string} value - The string to expand.
 * @returns {string} The expanded string.
 */
export function expandEnvVars(value: string): string {
    if (typeof value !== 'string') return value;
    // Handles ${VAR} format
    return value.replace(/\${([^}]+)}/g, (_, name) => EnvService.get(name) || "");
}

/**
 * Loads and parses an MCP bundle file, expanding environment variables.
 * Aligned with 2.1.19 gold reference for plugin/bundle loading.
 * 
 * @param {string} path - The absolute path to the bundle file.
 * @returns {Promise<any>} The parsed and expanded bundle content.
 */
export async function loadMcpBundle(path: string): Promise<any> {
    if (!existsSync(path)) {
        console.warn(`[MCP] Bundle not found at: ${path}`);
        return {};
    }

    try {
        console.log(`[MCP] Loading bundle from: ${path}`);
        const rawContent = readFileSync(path, 'utf8');
        const expandedContent = expandEnvVars(rawContent);

        const bundle = JSON.parse(expandedContent);
        // Returns either the full bundle or the mcpServers subset
        return bundle.mcpServers || bundle;
    } catch (error) {
        console.error(`[MCP] Failed to load or parse bundle at ${path}:`, error);
        return {};
    }
}
