/**
 * File: src/services/mcp/ToolSearchService.ts
 * Role: Manages deferred loading of MCP tools and provides the ToolSearch tool.
 */

import { EnvService } from "../config/EnvService.js";
import { terminalLog } from "../../utils/shared/runtime.js";

/**
 * Service to handle tool search and deferred loading logic.
 */
export class ToolSearchService {
    /**
     * Estimated token count for a tool collection.
     * Uses a simple character-based heuristic: total chars / 4.
     */
    static estimateTokenCount(tools: any[]): number {
        let totalChars = 0;
        for (const tool of tools) {
            totalChars += (tool.name || "").length;
            totalChars += (tool.description || "").length;
            if (tool.input_schema) {
                totalChars += JSON.stringify(tool.input_schema).length;
            }
        }
        return Math.floor(totalChars / 4);
    }

    /**
     * Identifies models that support tool references.
     * Based on reference implementation, Sonnet 3.5+ and Opus 3.5+ support this.
     */
    static supportsToolReference(model: string): boolean {
        const m = model.toLowerCase();
        // Check for Sonnet 3.5 or newer, or Opus
        return (m.includes("sonnet") && (m.includes("3-5") || m.includes("3.5") || m.includes("-latest"))) ||
            m.includes("opus") ||
            m.includes("claude-4");
    }

    /**
     * Determines if tool search should be enabled for the current request.
     */
    static async shouldEnableToolSearch(model: string, mcpTools: any[]): Promise<boolean> {
        const setting = EnvService.get("ENABLE_TOOL_SEARCH") || "auto";

        if (setting === "false") return false;
        if (setting === "true") return true;

        if (!this.supportsToolReference(model)) {
            terminalLog(`Tool search disabled for model '${model}': model does not support tool_reference blocks.`, "debug");
            return false;
        }

        const estimatedTokens = this.estimateTokenCount(mcpTools);
        // Default context window assumption if not provided (200k for Claude 3.5 Sonnet)
        const contextWindow = 200000;
        const threshold = EnvService.getToolSearchThreshold(contextWindow);

        const enabled = estimatedTokens >= threshold;

        if (enabled) {
            terminalLog(`Auto tool search enabled: ${estimatedTokens} estimated tokens (threshold: ${threshold})`, "info");
        } else {
            terminalLog(`Auto tool search disabled: ${estimatedTokens} estimated tokens (threshold: ${threshold})`, "debug");
        }

        return enabled;
    }

    /**
     * Returns the ToolSearch tool implementation.
     */
    static getToolSearchTool(allMcpTools: any[]): any {
        return {
            name: "ToolSearch",
            description: "Search for deferred tools when many tools are available. Use 'select:<tool_name>' for direct selection, or keywords to search.",
            input_schema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Query to find deferred tools. Use 'select:<tool_name>' for direct selection, or keywords to search."
                    },
                    max_results: {
                        type: "number",
                        description: "Maximum number of results to return (default: 5)",
                        default: 5
                    }
                },
                required: ["query"]
            },
            call: async (input: { query: string; max_results?: number }) => {
                const { query, max_results = 5 } = input;
                const mcpTools = allMcpTools.filter(t => t.isMcp);

                terminalLog(`ToolSearch called with query: "${query}"`, "info");

                // Direct selection
                const selectMatch = query.match(/^select:(.+)$/i);
                if (selectMatch) {
                    const targetName = selectMatch[1].trim();
                    const tool = mcpTools.find(t => t.name === targetName);
                    if (tool) {
                        return {
                            matches: [tool.name],
                            query,
                            total_deferred_tools: mcpTools.length
                        };
                    }
                    return { matches: [], query, total_deferred_tools: mcpTools.length };
                }

                // Simple keyword search
                const keywords = query.toLowerCase().split(/\s+/).filter(k => k.length > 0);
                const results = mcpTools
                    .map(tool => {
                        let score = 0;
                        const name = tool.name.toLowerCase();
                        const desc = (tool.description || "").toLowerCase();

                        for (const kw of keywords) {
                            if (name.includes(kw)) score += 10;
                            if (desc.includes(kw)) score += 5;
                        }
                        return { name: tool.name, score };
                    })
                    .filter(r => r.score > 0)
                    .sort((a, b) => b.score - a.score)
                    .slice(0, max_results)
                    .map(r => r.name);

                // Format as tool_reference markers as expected by discovery logic
                return results.map(name => ({
                    type: "tool_reference",
                    tool_name: name
                }));
            },
            // Metadata for ConversationService to handle tool_reference output
            isToolSearch: true
        };
    }

    /**
     * Scans message history for tool_reference markers to identify 'discovered' tools.
     */
    static getDiscoveredTools(messages: any[]): Set<string> {
        const discovered = new Set<string>();
        for (const msg of messages) {
            if (msg.role !== "user" || !Array.isArray(msg.content)) continue;

            for (const block of msg.content) {
                if (block.type === "tool_result" && Array.isArray(block.content)) {
                    for (const item of block.content) {
                        if (typeof item === "object" && item !== null && item.type === "tool_reference" && typeof item.tool_name === "string") {
                            discovered.add(item.tool_name);
                        }
                    }
                }
            }
        }
        return discovered;
    }
}
