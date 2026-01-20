
import { McpClient } from "./McpClient.js";
import { StdioMcpTransport } from "./McpTransport.js";
import { SseMcpTransport } from "./SseMcpTransport.js";
import { WebSocketMcpTransport } from "./WebSocketMcpTransport.js";
import { WebSocket } from "ws";
import { InitializeResultSchema } from "./McpSchemas.js";

// Logic for mergeMcpHeaders
async function mergeMcpHeaders(name: string, config: any) {
    const headers: Record<string, string> = { ...config.headers };
    // Potentially add auth headers or other config-specific headers here
    return headers;
}

export async function createMcpClient(name: string, config: any, options: any = {}) {
    let transport: any;

    try {
        if (config.type === "stdio" || !config.type) {
            const command = process.env.CLAUDE_CODE_SHELL_PREFIX || config.command;
            const args = process.env.CLAUDE_CODE_SHELL_PREFIX ?
                [[config.command, ...config.args].join(" ")] : config.args;

            transport = new StdioMcpTransport({
                command: command,
                args: args,
                env: { ...process.env, ...config.env },
                stderr: "pipe"
            });
        } else if (config.type === "sse" || config.type === "sse-ide") {
            const url = new URL(config.url);
            transport = new SseMcpTransport(url, {
                requestInit: {
                    headers: await mergeMcpHeaders(name, config)
                }
            });
        } else if (config.type === "ws" || config.type === "ws-ide") {
            const ws = new WebSocket(config.url);
            transport = new WebSocketMcpTransport(ws);
        } else {
            throw new Error(`Unsupported transport: ${config.type}`);
        }

        const client = new McpClient({
            taskStore: options.taskStore,
            taskMessageQueue: options.taskMessageQueue
        });

        // Set up request handler for roots/list before connecting
        client.setRequestHandler("roots/list", async () => {
            const { getOriginalCwd } = await import("../session/sessionStore.js");
            return {
                roots: [{
                    uri: `file://${getOriginalCwd()}`
                }]
            };
        });

        // connection with timeout
        const connectionPromise = client.connect(transport);
        const timeoutMs = 15000; // 15s timeout
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`Connection to MCP server "${name}" timed out after ${timeoutMs}ms`)), timeoutMs);
        });

        await Promise.race([connectionPromise, timeoutPromise]);

        // Handshake
        const initializeResult = await client.request({
            method: "initialize",
            params: {
                protocolVersion: "2024-11-05", // Updated based on spec
                capabilities: {
                    roots: { listChanged: true },
                    sampling: {}
                },
                clientInfo: {
                    name: "claude-code",
                    version: "2.0.76"
                }
            }
        }, InitializeResultSchema);

        await client.notification("notifications/initialized", {});

        // Setup error handlers
        client.onerror = (error) => {
            // Log connection errors
            console.error(`MCP Client ${name} error:`, error);
        };

        return {
            name,
            client,
            type: "connected",
            capabilities: initializeResult.capabilities || {},
            serverInfo: initializeResult.serverInfo,
            config,
            cleanup: async () => {
                try {
                    await client.close();
                    if (config.type === "stdio" && transport?.pid) {
                        // Ensure process is killed
                        try { process.kill(transport.pid, 'SIGINT'); } catch { }
                        // Fallback to SIGKILL if needed is handled by OS usually but could add more logic if needed
                    }
                } catch (e) {
                    console.error(`Error closing MCP client ${name}:`, e);
                }
            }
        };

    } catch (error) {
        return {
            name,
            type: "failed",
            error: error,
            config
        };
    }
}
