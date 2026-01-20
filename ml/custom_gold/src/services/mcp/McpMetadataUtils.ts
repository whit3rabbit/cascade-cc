
/**
 * Utilities for MCP server and tool metadata.
 * Logic from chunk_591.ts
 */

import { join } from "node:path";
import { existsSync } from "node:fs";
import { homedir } from "node:os";

/**
 * Normalizes an MCP server name.
 * Deobfuscated from B7 in chunk_591:433
 */
export function normalizeMcpServerName(name: string): string {
    return name.replace(/[^a-zA-Z0-9_-]/g, "_");
}

/**
 * Formats a tool name for an MCP server.
 */
export function formatMcpToolName(serverName: string, toolName: string): string {
    return `mcp__${normalizeMcpServerName(serverName)}__${toolName}`;
}

/**
 * Parses an MCP tool name back into server and tool components.
 * Deobfuscated from jF in chunk_591:71
 */
export function parseMcpToolName(fullName: string): { serverName: string; toolName?: string } | null {
    const parts = fullName.split("__");
    const [prefix, serverName, ...toolParts] = parts;

    if (prefix !== "mcp" || !serverName) return null;

    const toolName = toolParts.length > 0 ? toolParts.join("__") : undefined;
    return { serverName, toolName };
}

/**
 * Checks if a tool name is an MCP tool.
 * Deobfuscated from J_ in chunk_591:67
 */
export function isMcpTool(tool: { name?: string; isMcp?: boolean }): boolean {
    return !!(tool.name?.startsWith("mcp__") || tool.isMcp === true);
}

/**
 * Returns a human-readable description for an MCP scope.
 * Deobfuscated from r4A in chunk_591:124
 */
export function getMcpScopeDescription(scope: string): string {
    switch (scope) {
        case "local":
            return "Local config (private to you in this project)";
        case "project":
            return "Project config (shared via .mcp.json)";
        case "user":
            return "User config (available in all your projects)";
        case "dynamic":
            return "Dynamic config (from command line)";
        case "enterprise":
            return "Enterprise config (managed by your organization)";
        case "claudeai":
            return "claude.ai config";
        default:
            return scope;
    }
}

/**
 * Returns the path to the MCP CLI directory.
 * Deobfuscated from Bp in chunk_591:449
 */
export function getMcpCliDir(): string {
    return process.env.USE_MCP_CLI_DIR || join(homedir(), ".claude", "mcp-cli");
}
