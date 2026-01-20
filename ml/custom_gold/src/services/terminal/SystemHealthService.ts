
// Logic from chunk_543.ts (System Health Checks)

// --- Limits ---
const CLAUDE_MD_LIMIT = 50000; // Est. 50k chars
const AGENT_DESC_LIMIT = 10000; // Est. 10k tokens
const MCP_TOOL_LIMIT = 25000; // Est. 25k tokens

/**
 * Runs system health checks to identify potential performance blockers.
 */
export async function runSystemHealthCheck(options: {
    claudeMdFiles: any[],
    activeAgents: any[],
    mcpTools: any[]
}) {
    const { claudeMdFiles, activeAgents, mcpTools } = options;
    const warnings: any[] = [];

    // 1. Check CLAUDE.md size
    const largeFiles = claudeMdFiles.filter(f => f.content.length > CLAUDE_MD_LIMIT);
    if (largeFiles.length > 0) {
        warnings.push({
            type: "claude_md_size",
            message: `Large CLAUDE.md file detected (${largeFiles[0].content.length} chars)`,
            severity: "warning"
        });
    }

    // 2. Check Agent descriptions size
    const agentTokens = activeAgents.reduce((sum, agent) => sum + (agent.description?.length || 0) / 4, 0); // rough est
    if (agentTokens > AGENT_DESC_LIMIT) {
        warnings.push({
            type: "agent_desc_size",
            message: `Large agent descriptions context (~${Math.round(agentTokens)} tokens)`,
            severity: "warning"
        });
    }

    // 3. Check MCP Tool context size
    const mcpTokens = mcpTools.reduce((sum, tool) => sum + (tool.description?.length || 0) / 4, 0); // rough est
    if (mcpTokens > MCP_TOOL_LIMIT) {
        warnings.push({
            type: "mcp_tool_size",
            message: `Large MCP tools context (~${Math.round(mcpTokens)} tokens)`,
            severity: "warning"
        });
    }

    return {
        hasWarnings: warnings.length > 0,
        warnings
    };
}
