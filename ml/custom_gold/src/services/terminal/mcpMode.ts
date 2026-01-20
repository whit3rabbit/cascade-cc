/**
 * Detects the current tool execution mode (TST, MCP-CLI, or standard).
 */

function isEnvEnabled(value: string | undefined): boolean {
    return value === "true" || value === "1";
}

function isEnvDisabled(value: string | undefined): boolean {
    return value === "false" || value === "0";
}

export type McpMode = "tst" | "mcp-cli" | "standard";

export function getMcpMode(): McpMode {
    if (isEnvEnabled(process.env.ENABLE_TOOL_SEARCH)) return "tst";
    if (isEnvEnabled(process.env.ENABLE_MCP_CLI)) return "mcp-cli";
    if (isEnvDisabled(process.env.ENABLE_MCP_CLI)) return "standard";
    if (isEnvDisabled(process.env.ENABLE_TOOL_SEARCH)) return "standard";
    return "tst"; // Default to TST if not specified? 
}

export function getExternalMcpMode(): McpMode {
    if (isEnvEnabled(process.env.ENABLE_TOOL_SEARCH)) return "tst";
    if (isEnvEnabled(process.env.ENABLE_EXPERIMENTAL_MCP_CLI)) return "mcp-cli";
    if (isEnvDisabled(process.env.ENABLE_TOOL_SEARCH)) return "standard";
    if (isEnvDisabled(process.env.ENABLE_EXPERIMENTAL_MCP_CLI)) return "standard";

    // Fallback to statsig if available
    // try {
    //     if (getStatsigParam("tengu_mcp_tool_search", false)) return "tst";
    // } catch {}

    return "standard";
}

export function isToolSearchEnabled(): boolean {
    return getExternalMcpMode() === "tst";
}

export function isToolReferenceBlock(block: any): boolean {
    return typeof block === "object" && block !== null && "type" in block && block.type === "tool_reference";
}
