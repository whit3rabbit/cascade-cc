import { McpServerManager } from "./McpServerManager.js";
import { getMcpClient } from "./McpClientManager.js";
import { fetchMcpTools, fetchMcpResources } from "./McpDiscovery.js";
import { log } from "../logger/loggerService.js";
import { z } from "zod";

const logger = log("mcp-cli");

function parseToolId(toolId: string) {
    const parts = toolId.split("/");
    if (parts.length < 2) throw new Error("Invalid tool ID format. Expected <server>/<tool>");
    return { server: parts[0], tool: parts.slice(1).join("/") };
}

export const McpCliService = {
    async listServers(options: { json?: boolean } = {}) {
        const { servers } = await McpServerManager.getAllMcpServers();
        const result = Object.entries(servers).map(([name, config]) => ({
            name,
            type: (config as any).type,
            status: "configured" // We'd need live status for more accuracy
        }));

        if (options.json) {
            console.log(JSON.stringify(result, null, 2));
        } else {
            if (result.length === 0) {
                console.log("No MCP servers configured.");
            } else {
                result.forEach(s => console.log(`${s.name} (${s.type})`));
            }
        }
    },

    async listTools(options: { json?: boolean, server?: string } = {}) {
        const { servers } = await McpServerManager.getAllMcpServers();
        const allTools: any[] = [];

        for (const [name, config] of Object.entries(servers)) {
            if (options.server && name !== options.server) continue;
            try {
                const client = await getMcpClient(name, config);
                if (client.type === 'connected') {
                    const tools = await fetchMcpTools(client);
                    allTools.push(...tools.map((t: any) => ({
                        ...t,
                        server: name,
                        id: `${name}/${t.originalName || t.name.split('__').pop()}`
                    })));
                }
            } catch (err) {
                // Ignore errors, just skip
            }
        }

        if (options.json) {
            console.log(JSON.stringify(allTools, null, 2));
        } else {
            if (allTools.length === 0) {
                console.log("No tools found.");
            } else {
                allTools.forEach(t => {
                    console.log(`${t.id}${t.description ? ` - ${t.description.split('\n')[0].substring(0, 50)}...` : ''}`);
                });
            }
        }
    },

    async callTool(toolId: string, args: string, options: { timeout?: number, json?: boolean, debug?: boolean } = {}) {
        const { server, tool } = parseToolId(toolId);

        if (options.debug) console.error(`Connecting to ${server}...`);

        const { servers } = await McpServerManager.getAllMcpServers();
        const config = servers[server];
        if (!config) throw new Error(`Server '${server}' not found`);

        const clientWrapper = await getMcpClient(server, config);

        if (clientWrapper.type !== 'connected') {
            throw new Error(`Failed to connect to server '${server}' (status: ${clientWrapper.type})`);
        }

        if (options.debug) console.error(`Calling tool ${tool}...`);

        // Find the actual internal tool name if needed (handling prefixing)
        // fetchMcpTools adds prefix, but we need to call with correct name expected by server?
        // Actually fetchMcpTools wraps the call, but here we are calling client.request direct.
        // The server expects the original tool name.

        let toolName = tool;
        // Verify tool exists
        const tools = await fetchMcpTools(clientWrapper);
        const toolDef = tools.find((t: any) => t.originalName === tool || t.name.endsWith(`__${tool}`));
        if (!toolDef) throw new Error(`Tool '${tool}' not found on server '${server}'`);

        // If fetchMcpTools returns wrapped definitions, strict usage implies using original name for the actual direct call
        toolName = toolDef.originalName || tool;

        const timeout = options.timeout || 300000; // 5 min default

        try {
            let parsedArgs;
            try {
                parsedArgs = JSON.parse(args);
            } catch (e) {
                throw new Error("Arguments must be valid JSON");
            }

            const result = await clientWrapper.client.request(
                {
                    method: "tools/call",
                    params: {
                        name: toolName,
                        arguments: parsedArgs
                    }
                },
                z.any(), // Generic schema
                { timeout }
            );

            if (options.json) {
                console.log(JSON.stringify(result, null, 2));
            } else {
                // Basic formatting
                if (result.content) {
                    result.content.forEach((item: any) => {
                        if (item.type === 'text') console.log(item.text);
                        else console.log(`[${item.type}] ${JSON.stringify(item)}`);
                    });
                } else {
                    console.log(JSON.stringify(result, null, 2));
                }
            }
            return result;
        } catch (error) {
            console.error(`Error calling tool: ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }
    },

    async getTool(toolId: string, options: { json?: boolean } = {}) {
        const { server, tool } = parseToolId(toolId);
        const { servers } = await McpServerManager.getAllMcpServers();
        const config = servers[server];

        if (!config) throw new Error(`Server '${server}' not found`);
        const clientWrapper = await getMcpClient(server, config);

        if (clientWrapper.type !== 'connected') throw new Error(`Server '${server}' is not connected`);

        const tools = await fetchMcpTools(clientWrapper);
        const found = tools.find((t: any) => t.originalName === tool || t.name.endsWith(`__${tool}`));

        if (!found) throw new Error(`Tool '${tool}' not found on server '${server}'`);

        if (options.json) {
            console.log(JSON.stringify(found, null, 2));
        } else {
            console.log(`Tool: ${toolId}`);
            console.log(`Description: ${found.description || 'N/A'}`);
            console.log("Input Schema:");
            console.log(JSON.stringify(found.inputSchema, null, 2));
        }
        return found;
    },

    async listResources(serverName?: string, options: { json?: boolean } = {}) {
        const { servers } = await McpServerManager.getAllMcpServers();
        const allResources: any[] = [];

        for (const [name, config] of Object.entries(servers)) {
            if (serverName && name !== serverName) continue;
            try {
                const client = await getMcpClient(name, config);
                if (client.type === 'connected') {
                    const resources = await fetchMcpResources(client);
                    allResources.push(...resources.map((r: any) => ({
                        ...r,
                        server: name
                    })));
                }
            } catch (err) {
                // Ignore
            }
        }

        if (options.json) {
            console.log(JSON.stringify(allResources, null, 2));
        } else {
            if (allResources.length === 0) {
                console.log("No resources found.");
            } else {
                allResources.forEach(r => {
                    console.log(`${r.server}/${r.name || r.uri} (${r.mimeType || 'unknown'})`);
                });
            }
        }
    }
};
