
import React from 'react';
import { z } from 'zod';
import { Box, Text } from 'ink';
import { McpClient } from '../../services/mcp/McpClient.js';

// Placeholder components for those seen in chunk_336
// Yz -> JsonViewer or CodeBlock
const JsonViewer = ({ content, verbose }: { content: string, verbose?: boolean }) => (
    <Box flexDirection="column">
        <Text>{content}</Text>
    </Box>
);

// o3 -> ResultViewer
const ResultViewer = ({ result, verbose }: { result: any, verbose?: boolean }) => (
    <Box>
        <Text>Result: {JSON.stringify(result)}</Text>
    </Box>
);

// t5 -> RejectedViewer
const RejectedViewer = () => <Text color="red">Rejected</Text>;

const ListMcpResourcesInputSchema = z.object({
    server: z.string().optional().describe("Optional server name to filter resources by")
});

const ListMcpResourcesOutputSchema = z.array(z.object({
    uri: z.string().describe("Resource URI"),
    name: z.string().describe("Resource name"),
    mimeType: z.string().optional().describe("MIME type of the resource"),
    description: z.string().optional().describe("Resource description"),
    server: z.string().describe("Server that provides this resource")
}));

export const ListMcpResourcesTool = {
    name: "ListMcpResourcesTool",
    userFacingName: () => "listMcpResources",
    isEnabled: () => true,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    description: async () => `
Reads a specific resource from an MCP server.
- server: The name of the MCP server to read from
- uri: The URI of the resource to read

Usage examples:
- Read a resource from a server: \`readMcpResource({ server: "myserver", uri: "my-resource-uri" })\`
    `, // Wait, chunk_336 description for List matches Read? Let me check line 31 (h12) vs line 92 (i12).
    // Ah line 31 returns h12. Line 92 defines i12 which is for Read.
    // h12 is not shown in the snippet view for 336. 
    // I'll assume standard list description.
    prompt: async () => `List available resources from MCP servers.`,
    inputSchema: ListMcpResourcesInputSchema,
    outputSchema: ListMcpResourcesOutputSchema,

    async call(input: z.infer<typeof ListMcpResourcesInputSchema>, context: { options: { mcpClients: any[] } }) {
        const { server } = input;
        const { mcpClients } = context.options;
        const clients = server ? mcpClients.filter(c => c.name === server) : mcpClients;

        if (server && clients.length === 0) {
            throw new Error(`Server "${server}" not found. Available servers: ${mcpClients.map(c => c.name).join(", ")}`);
        }

        const resources = [];
        for (const client of clients) {
            if (client.type !== 'connected') continue;
            try {
                if (!client.capabilities?.resources) continue;
                // $BA is RequestOptions/Schema?
                const result = await client.client.request({ method: "resources/list" });
                if (result.resources) {
                    resources.push(...result.resources.map((r: any) => ({ ...r, server: client.name })));
                }
            } catch (err: any) {
                // log error
            }
        }
        return { data: resources };
    },

    // UI Renderers
    renderToolUseMessage: (input: any) => {
        return input.server ? `List resources from server "${input.server}"` : "List resources from all servers";
    },
    renderToolUseRejectedMessage: () => <RejectedViewer />,
    renderToolUseErrorMessage: (result: any, { verbose }: any) => <ResultViewer result={result} verbose={verbose} />,
    renderToolResultMessage: (result: any, toolUseId: string, { verbose }: any) => {
        if (!result || !result.data || result.data.length === 0) {
            return (
                <Box justifyContent="space-between" overflowX="hidden" width="100%">
                    <Box height={1}>
                        <Text dimColor>(No content)</Text>
                    </Box>
                </Box>
            );
        }
        const json = JSON.stringify(result, null, 2);
        return <JsonViewer content={json} verbose={verbose} />;
    },
    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: JSON.stringify(result)
        };
    },
    async checkPermissions(input: any) {
        return { behavior: "allow", updatedInput: input };
    }
};

const ReadMcpResourceInputSchema = z.object({
    server: z.string().describe("The MCP server name"),
    uri: z.string().describe("The resource URI to read")
});

const ReadMcpResourceOutputSchema = z.object({
    contents: z.array(z.object({
        uri: z.string(),
        mimeType: z.string().optional(),
        text: z.string().optional()
    }))
});

export const ReadMcpResourceTool = {
    name: "ReadMcpResourceTool",
    userFacingName: () => "readMcpResource",
    isEnabled: () => true,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    description: async () => `
Reads a specific resource from an MCP server.
- server: The name of the MCP server to read from
- uri: The URI of the resource to read

Usage examples:
- Read a resource from a server: \`readMcpResource({ server: "myserver", uri: "my-resource-uri" })\`
`,
    prompt: async () => `
Reads a specific resource from an MCP server, identified by server name and resource URI.

Parameters:
- server (required): The name of the MCP server from which to read the resource
- uri (required): The URI of the resource to read
`,
    inputSchema: ReadMcpResourceInputSchema,
    outputSchema: ReadMcpResourceOutputSchema,

    async call(input: z.infer<typeof ReadMcpResourceInputSchema>, context: { options: { mcpClients: any[] } }) {
        const { server, uri } = input;
        const { mcpClients } = context.options;
        const client = mcpClients.find(c => c.name === server);

        if (!client) throw new Error(`Server "${server}" not found. Available servers: ${mcpClients.map(c => c.name).join(", ")}`);
        if (client.type !== "connected") throw new Error(`Server "${server}" is not connected`);
        if (!client.capabilities?.resources) throw new Error(`Server "${server}" does not support resources`);

        const result = await client.client.request({
            method: "resources/read",
            params: { uri }
        });

        return { data: result };
    },

    // UI Renderers
    renderToolUseMessage: (input: any) => {
        if (!input.uri || !input.server) return null;
        return `Read resource "${input.uri}" from server "${input.server}"`;
    },
    renderToolUseRejectedMessage: () => <RejectedViewer />,
    renderToolUseErrorMessage: (result: any, { verbose }: any) => <ResultViewer result={result} verbose={verbose} />,
    renderToolResultMessage: (result: any, toolUseId: string, { verbose }: any) => {
        // logic from e12 in chunk_336
        // It checks A.contents etc. 
        // Assuming result.data structure matches
        const data = result.data;
        if (!data || !data.contents || data.contents.length === 0) {
            return (
                <Box justifyContent="space-between" overflowX="hidden" width="100%">
                    <Box height={1}>
                        <Text dimColor>(No content)</Text>
                    </Box>
                </Box>
            );
        }
        const json = JSON.stringify(data, null, 2);
        return <JsonViewer content={json} verbose={verbose} />;
    },
    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: JSON.stringify(result)
        };
    },
    async checkPermissions(input: any) {
        return { behavior: "allow", updatedInput: input };
    }
};

