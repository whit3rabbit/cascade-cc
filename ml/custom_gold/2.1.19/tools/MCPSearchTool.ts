
import { mcpClientManager } from '../services/mcp/McpClientManager.js';
import { McpServerManager } from '../services/mcp/McpServerManager.js';

export interface MCPSearchInput {
    query: string;
}

export const MCPSearchTool = {
    name: "MCPSearch",
    description: "Search for MCP tools and resources across all connected servers.",
    async call(input: MCPSearchInput) {
        const { query } = input;
        const results: string[] = [];
        const servers = await McpServerManager.getAllMcpServers();

        for (const serverName of Object.keys(servers)) {
            const client = mcpClientManager.getClient(serverName);
            if (!client) continue;

            try {
                // Search tools
                const tools = await client.listTools();
                const matchedTools = tools.tools.filter(t =>
                    t.name.toLowerCase().includes(query.toLowerCase()) ||
                    (t.description && t.description.toLowerCase().includes(query.toLowerCase()))
                );

                matchedTools.forEach(t => {
                    results.push(`[Tool] ${serverName}/${t.name}: ${t.description?.slice(0, 100)}`);
                });

                // Search resources
                const resources = await client.listResources();
                const matchedResources = resources.resources.filter(r =>
                    r.name.toLowerCase().includes(query.toLowerCase()) ||
                    r.uri.toLowerCase().includes(query.toLowerCase())
                );

                matchedResources.forEach(r => {
                    results.push(`[Resource] ${serverName}/${r.name} (${r.uri})`);
                });

            } catch (e) {
                // ignore failed searches on specific servers
            }
        }

        if (results.length === 0) {
            return {
                is_error: false,
                content: "No matching MCP tools or resources found."
            };
        }

        return {
            is_error: false,
            content: results.join('\n')
        };
    }
};
