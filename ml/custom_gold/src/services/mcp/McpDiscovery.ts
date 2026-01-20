import { callMcpTool } from "./McpToolExecution.js";
import { connectAllMcpServers } from "./McpClientManager.js";
import { ListToolsResultSchema, ListResourcesResultSchema, ListPromptsResultSchema } from "./McpSchemas.js";

export async function fetchMcpTools(clientWrapper: any) {
    // Logic from JKA
    if (clientWrapper.type !== "connected") return [];
    try {
        if (!clientWrapper.capabilities?.tools) return [];
        const result = await clientWrapper.client.request({ method: "tools/list" }, ListToolsResultSchema);

        return result.tools.map((tool: any) => ({
            name: `mcp__${clientWrapper.name}__${tool.name}`,
            originalName: tool.name,
            description: tool.description,
            inputSchema: tool.inputSchema,
            call: async (args: any) => {
                return callMcpTool(clientWrapper, tool.name, args);
            }
        }));
    } catch (e) {
        return [];
    }
}

export async function fetchMcpResources(clientWrapper: any) {
    // Logic from T82
    if (clientWrapper.type !== "connected") return [];
    try {
        const result = await clientWrapper.client.request({ method: "resources/list" }, ListResourcesResultSchema);
        return result.resources || [];
    } catch { return []; }
}

export async function fetchMcpPrompts(clientWrapper: any) {
    // Logic from P82
    if (clientWrapper.type !== "connected") return [];
    try {
        const result = await clientWrapper.client.request({ method: "prompts/list" }, ListPromptsResultSchema);
        return result.prompts || [];
    } catch { return []; }
}

export async function fetchAllMcpData() {
    // Logic from n51
    const clients: any[] = [];
    const tools: any[] = [];
    const commands: any[] = [];

    await new Promise<void>((resolve) => {
        let count = 0;
        // This relies on knowing total count ahead of time, or just waiting for completion?
        // n51 uses a callback strategy.
        // We'll treat connectAllMcpServers as async and wait.
        connectAllMcpServers(async (clientWrapper) => {
            clients.push(clientWrapper);
            const clientTools = await fetchMcpTools(clientWrapper);
            const clientPrompts = await fetchMcpPrompts(clientWrapper);
            // resources?
            tools.push(...clientTools);
            commands.push(...clientPrompts);
        }).then(() => resolve());
    });

    return { clients, tools, commands };
}
