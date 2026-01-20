import React from "react";
import { Box, Text } from "ink";
import { Select } from "../../components/shared/Select.js";
import { createServer, Server, IncomingMessage, ServerResponse } from "node:http";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { join } from "node:path";
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { z } from "zod";
import { log } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { getConfigDir } from "../../utils/shared/pathUtils.js";
import { homedir } from 'node:os';

const logger = log("mcp-cli-endpoint");

/**
 * Schemas for local CLI endpoint protocol
 */
const ServerInfoSchema = z.object({
    name: z.string(),
    type: z.string(),
    hasTools: z.boolean().optional(),
    hasResources: z.boolean().optional(),
    hasPrompts: z.boolean().optional(),
    serverInfo: z.object({
        name: z.string(),
        version: z.string()
    }).optional()
});

const ToolsRequestSchema = z.object({
    command: z.literal("tools"),
    params: z.object({
        server: z.string().optional()
    }).optional()
});

const CallToolRequestSchema = z.object({
    command: z.literal("call"),
    params: z.object({
        server: z.string(),
        tool: z.string(),
        args: z.record(z.string(), z.unknown()),
        timeoutMs: z.number().optional()
    })
});

const ReadResourceRequestSchema = z.object({
    command: z.literal("read"),
    params: z.object({
        server: z.string(),
        uri: z.string(),
        timeoutMs: z.number().optional()
    })
});

const ServersRequestSchema = z.object({ command: z.literal("servers") });
const InfoRequestSchema = z.object({
    command: z.literal("info"),
    params: z.object({ server: z.string(), toolName: z.string() })
});
const GrepRequestSchema = z.object({
    command: z.literal("grep"),
    params: z.object({ pattern: z.string(), ignoreCase: z.boolean().optional() })
});
const ResourcesRequestSchema = z.object({
    command: z.literal("resources"),
    params: z.object({ server: z.string().optional() }).optional()
});

const CommandSchema = z.discriminatedUnion("command", [
    ServersRequestSchema,
    ToolsRequestSchema,
    CallToolRequestSchema,
    ReadResourceRequestSchema,
    InfoRequestSchema,
    GrepRequestSchema,
    ResourcesRequestSchema
]);

function getEndpointPath() {
    return join(getConfigDir(), `claude-code-${process.pid}.endpoint`);
}

/**
 * MCP CLI Endpoint manager.
 * Provides a remote interface for other tools (like the CLI in another terminal) to query tools/resources.
 * Logic from chunk_592.ts
 */
export class McpCliEndpoint {
    private server: Server | null = null;
    private secret: string;
    private port: number | null = null;
    private mcpClients: any[] = [];
    private availableTools: any[] = [];
    private resources: Record<string, any[]> = {};

    constructor(mcpClients: any[], availableTools: any[]) {
        this.mcpClients = mcpClients;
        this.availableTools = availableTools || [];
        this.secret = randomBytes(32).toString("hex");
    }

    async start(): Promise<{ port: number; url: string }> {
        if (this.server) throw new Error("MCP CLI endpoint already started");

        return new Promise((resolve, reject) => {
            this.server = createServer((req, res) => {
                this.handleRequest(req, res);
            });

            this.server.on("error", (err) => {
                logger.error("Server error", err);
                reject(err);
            });

            this.server.listen(0, "127.0.0.1", () => {
                const addr = this.server?.address();
                if (!addr || typeof addr === "string") {
                    reject(new Error("Failed to get server address"));
                    return;
                }
                this.port = addr.port;
                const url = `http://127.0.0.1:${this.port}`;
                logger.info(`[MCP CLI Endpoint] Started on ${url}`);
                this.writeEndpointFile(url, this.secret);
                resolve({ port: this.port, url });
            });
        });
    }

    private writeEndpointFile(url: string, key: string) {
        try {
            const dir = getConfigDir();
            mkdirSync(dir, { recursive: true });
            const path = getEndpointPath();
            const data = JSON.stringify({ url, key });
            writeFileSync(path, Buffer.from(data).toString("base64"), { mode: 0o600 });
        } catch (err) {
            logger.error("Failed to write endpoint file", err);
        }
    }

    getSecret() {
        return this.secret;
    }

    updateClients(clients: any[]) {
        this.mcpClients = clients;
    }

    updateTools(tools: any[]) {
        this.availableTools = tools;
    }

    updateResources(resources: Record<string, any[]>) {
        this.resources = resources;
    }

    private getNormalizedNames() {
        const map: Record<string, string> = {};
        for (const client of this.mcpClients) {
            map[sanitizeMcpName(client.name)] = client.name;
        }
        return map;
    }

    private validateSecret(token: string): boolean {
        try {
            const a = Buffer.from(token);
            const b = Buffer.from(this.secret);
            if (a.length !== b.length) return false;
            return timingSafeEqual(a, b);
        } catch {
            return false;
        }
    }

    private async handleRequest(req: IncomingMessage, res: ServerResponse) {
        req.setTimeout(30000);

        if (req.method !== "POST" || req.url !== "/mcp") {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Not Found" }));
            return;
        }

        const auth = req.headers.authorization;
        if (!auth?.startsWith("Bearer ")) {
            res.writeHead(403, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Forbidden" }));
            return;
        }

        const token = auth.slice(7);
        if (!this.validateSecret(token)) {
            res.writeHead(403, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Forbidden" }));
            return;
        }

        let body = "";
        let size = 0;
        const limit = 10 * 1024 * 1024; // 10MB

        req.on("data", chunk => {
            size += chunk.length;
            if (size > limit) {
                res.writeHead(413, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: "Payload Too Large" }));
                req.destroy();
                return;
            }
            body += chunk;
        });

        req.on("end", async () => {
            try {
                const json = JSON.parse(body);
                const command = CommandSchema.parse(json);
                const result = await this.handleCommand(command);
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify(result));
            } catch (err: any) {
                const status = (err instanceof SyntaxError || err.name === "ZodError") ? 400 : 500;
                res.writeHead(status, { "Content-Type": "application/json" });
                res.end(JSON.stringify({
                    error: err.message || "Unknown Error",
                    type: err.constructor.name
                }));
                logger.error("Command handling error", err);
            }
        });
    }

    private async handleCommand(cmd: z.infer<typeof CommandSchema>) {
        const start = Date.now();
        try {
            const result = await this.executeCommand(cmd);
            const duration = Date.now() - start;

            logTelemetryEvent("tengu_mcp_cli_command_executed", {
                command: cmd.command,
                success: true,
                duration_ms: duration,
                ...(result as any).metadata
            });

            return result;
        } catch (err: any) {
            const duration = Date.now() - start;
            logTelemetryEvent("tengu_mcp_cli_command_executed", {
                command: cmd.command,
                success: false,
                error_type: err.constructor.name,
                duration_ms: duration
            });
            throw err;
        }
    }

    private async executeCommand(cmd: z.infer<typeof CommandSchema>) {
        switch (cmd.command) {
            case "servers": {
                return {
                    data: this.mcpClients.map(c => ({
                        name: c.name,
                        type: c.type,
                        hasTools: c.capabilities?.tools !== undefined,
                        hasResources: c.capabilities?.resources !== undefined,
                        hasPrompts: c.capabilities?.prompts !== undefined
                    })),
                    metadata: { count: this.mcpClients.length }
                };
            }
            case "tools":
                // TODO: Implement filtering properly using params.server
                return {
                    data: this.availableTools,
                    metadata: { count: this.availableTools.length }
                };
            case "call": {
                const { server, tool, args, timeoutMs } = cmd.params;
                const result = await this.callTool(server, tool, args, timeoutMs);
                return { data: result };
            }
            case "read": {
                const { server, uri, timeoutMs } = cmd.params;
                const result = await this.readResource(server, uri, timeoutMs);
                return { data: result };
            }
            case "resources": {
                const { server } = cmd.params || {};
                let result = Object.values(this.resources).flat();
                if (server) {
                    const normalized = sanitizeMcpName(server);
                    result = result.filter(r => sanitizeMcpName(r.server) === normalized);
                }
                return {
                    data: result,
                    metadata: { count: result.length }
                };
            }
            case "grep": {
                const { pattern, ignoreCase } = cmd.params;
                // Basic grep implementation using ripgrep
                const args = ["--json", "-e", pattern];
                if (ignoreCase) args.push("-i");
                // Search in current directory by default
                args.push(".");
                try {
                    const output = await runRipgrep(args);
                    // Parse ndjson output
                    const matches = output.trim().split('\n').map((line: string) => {
                        try { return JSON.parse(line); } catch { return null; }
                    }).filter((x: any) => x && x.type === 'match');
                    return { data: matches };
                } catch (e: any) {
                    return { error: e.message };
                }
            }
            case "info": {
                // Info about a tool or server
                return { data: "Not implemented" };
            }
            default:
                return { data: "Not implemented" };
        }
    }

    async callTool(server: string, tool: string, args: any, timeoutMs?: number) {
        const clientWrapper = this.getConnectedClient(server);
        if (!clientWrapper) throw new Error(`Server ${server} not found`);

        const toolName = `mcp__${sanitizeMcpName(server)}__${sanitizeMcpName(tool)}`;
        const toolDef = this.availableTools.find(t => t.name === toolName || t.name === tool); // support full name too

        const actualName = toolDef?.originalName || tool;

        return await clientWrapper.client.request(
            {
                method: "tools/call",
                params: {
                    name: actualName,
                    arguments: args
                }
            },
            z.any(),
            timeoutMs ? { signal: AbortSignal.timeout(timeoutMs) } : undefined
        );
    }

    async readResource(server: string, uri: string, timeoutMs?: number) {
        const clientWrapper = this.getConnectedClient(server);
        if (!clientWrapper) throw new Error(`Server ${server} not found`);

        return await clientWrapper.client.request(
            {
                method: "resources/read",
                params: {
                    uri: uri
                }
            },
            z.any(),
            timeoutMs ? { signal: AbortSignal.timeout(timeoutMs) } : undefined
        );
    }

    private getConnectedClient(server: string) {
        // Loose matching
        const normalized = sanitizeMcpName(server);
        const map = this.getNormalizedNames();
        const actualName = map[normalized];
        if (!actualName) return null;

        const client = this.mcpClients.find(c => c.name === actualName);
        if (client && client.type !== 'connected') {
            throw new Error(`Server '${server}' is not connected (${client.type})`);
        }
        return client;
    }

    async stop() {
        if (!this.server) return;
        return new Promise<void>((resolve, reject) => {
            this.server?.close((err) => {
                if (err) reject(err);
                else {
                    this.server = null;
                    resolve();
                }
            });
        });
    }
}

function sanitizeMcpName(name: string) {
    return name.replace(/[^a-zA-Z0-9_\-]/g, "_");
}

/**
 * Configuration Error Dialog (eI7)
 */
export function ConfigErrorDialog({ filePath, errorDescription, onExit, onReset }: any) {
    return (
        <Box flexDirection="column" borderColor="red" borderStyle="round" padding={1} width={80}>
            <Text bold color="red">Configuration Error</Text>
            <Box marginY={1} flexDirection="column">
                <Text>The configuration file at <Text bold>{filePath}</Text> contains invalid JSON.</Text>
                <Text>{errorDescription}</Text>
            </Box>
            <Box marginTop={1} flexDirection="column">
                <Text bold>Choose an option:</Text>
                <Select
                    options={[
                        { label: "Exit and fix manually", value: "exit" },
                        { label: "Reset with default configuration", value: "reset" }
                    ]}
                    onChange={(val: string) => val === "exit" ? onExit() : onReset()}
                />
            </Box>
        </Box>
    );
}

/**
 * Node.js ripgrep bridge (oI7).
 */
export async function runRipgrep(args: string[]): Promise<any> {
    const { spawn } = await import("child_process");
    return new Promise((resolve, reject) => {
        const rg = spawn("rg", args);
        let stdout = "";
        rg.stdout.on("data", d => stdout += d);
        rg.on("close", code => {
            resolve(stdout);
        });
        rg.on("error", reject);
    });
}
