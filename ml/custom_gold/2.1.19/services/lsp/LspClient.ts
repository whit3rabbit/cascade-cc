/**
 * File: src/services/lsp/LspClient.ts
 * Role: Low-level LSP client handling stdio communication with a language server.
 */

import { spawn, ChildProcess } from 'node:child_process';
import { LspServerConfig } from './LspServerManager.js';

export type LspState = "stopped" | "starting" | "running" | "error";

export class LspClient {
    public state: LspState = "stopped";
    private process?: ChildProcess;
    private requestId = 0;
    private pendingRequests: Map<number, { resolve: (val: any) => void, reject: (err: Error) => void }> = new Map();
    private requestHandlers: Map<string, (params: any) => any> = new Map();
    private buffer = "";

    constructor(public readonly name: string, public readonly config: LspServerConfig) { }

    async start(): Promise<void> {
        if (this.state === "running" || this.state === "starting") return;
        this.state = "starting";

        try {
            this.process = spawn(this.config.command, this.config.args, {
                env: { ...process.env, ...this.config.env },
                stdio: ['pipe', 'pipe', 'inherit']
            });

            this.process.on("error", (err) => {
                console.error(`[LSP:${this.name}] Process error:`, err);
                this.state = "error";
            });

            this.process.on("exit", (code) => {
                console.warn(`[LSP:${this.name}] Process exited with code ${code}`);
                this.state = "stopped";
            });

            this.process.stdout?.on("data", (chunk: Buffer) => {
                this.handleData(chunk.toString());
            });

            // Standard LSP initialize request
            const initResponse = await this.sendRequest("initialize", {
                processId: process.pid,
                rootUri: `file://${process.cwd()}`,
                capabilities: {},
                workspaceFolders: [{ uri: `file://${process.cwd()}`, name: "workspace" }]
            });

            await this.sendNotification("initialized", {});
            this.state = "running";
        } catch (error) {
            this.state = "error";
            throw error;
        }
    }

    async stop(): Promise<void> {
        if (this.process) {
            this.process.kill();
            this.process = undefined;
        }
        this.state = "stopped";
    }

    async sendRequest(method: string, params: any): Promise<any> {
        const id = ++this.requestId;
        const message = { jsonrpc: "2.0", id, method, params };
        const payload = this.frameMessage(message);

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });
            this.process?.stdin?.write(payload);
        });
    }

    async sendNotification(method: string, params: any): Promise<void> {
        const message = { jsonrpc: "2.0", method, params };
        const payload = this.frameMessage(message);
        this.process?.stdin?.write(payload);
    }

    onRequest(method: string, handler: (params: any) => any): void {
        this.requestHandlers.set(method, handler);
    }

    private frameMessage(message: any): string {
        const content = JSON.stringify(message);
        return `Content-Length: ${Buffer.byteLength(content, 'utf8')}\r\n\r\n${content}`;
    }

    private handleData(data: string): void {
        this.buffer += data;
        while (true) {
            const contentLengthMatch = this.buffer.match(/Content-Length: (\d+)\r\n\r\n/);
            if (!contentLengthMatch) break;

            const headerLength = contentLengthMatch[0].length;
            const contentLength = parseInt(contentLengthMatch[1], 10);

            if (this.buffer.length < headerLength + contentLength) break;

            const content = this.buffer.substring(headerLength, headerLength + contentLength);
            this.buffer = this.buffer.substring(headerLength + contentLength);

            try {
                const message = JSON.parse(content);
                this.dispatchMessage(message);
            } catch (err) {
                console.error(`[LSP:${this.name}] Failed to parse message:`, err);
            }
        }
    }

    private dispatchMessage(message: any): void {
        if (message.id !== undefined) {
            if (message.method) {
                // Request from server to client
                const handler = this.requestHandlers.get(message.method);
                if (handler) {
                    const result = handler(message.params);
                    this.sendResponse(message.id, result);
                }
            } else {
                // Response from server to client
                const pending = this.pendingRequests.get(message.id);
                if (pending) {
                    this.pendingRequests.delete(message.id);
                    if (message.error) pending.reject(new Error(message.error.message));
                    else pending.resolve(message.result);
                }
            }
        }
        // Notifications are ignored for now
    }

    private async sendResponse(id: number, result: any): Promise<void> {
        const message = { jsonrpc: "2.0", id, result };
        const payload = this.frameMessage(message);
        this.process?.stdin?.write(payload);
    }
}
