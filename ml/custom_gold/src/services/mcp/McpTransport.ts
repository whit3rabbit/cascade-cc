
import { spawn, ChildProcess } from "node:child_process";
import { JsonRpcMessage, JsonRpcMessageSchema } from "./McpSchemas.js";
import { Buffer } from "node:buffer";
import * as os from "node:os";

export interface McpTransport {
    start(): Promise<void>;
    close(): Promise<void>;
    send(message: JsonRpcMessage, options?: any): Promise<void>;
    onclose?: () => void;
    onerror?: (error: Error) => void;
    onmessage?: (message: JsonRpcMessage) => void;
}

export interface StdioTransportParams {
    command: string;
    args?: string[];
    env?: Record<string, string>;
    stderr?: "inherit" | "pipe" | "ignore";
    cwd?: string;
}

export class MessageBuffer {
    private _buffer?: Buffer;

    append(data: Buffer) {
        this._buffer = this._buffer ? Buffer.concat([this._buffer, data]) : data;
    }

    readMessage(): JsonRpcMessage | null {
        if (!this._buffer || this._buffer.length === 0) return null;

        const index = this._buffer.indexOf('\n');
        if (index === -1) return null;

        const line = this._buffer.toString("utf8", 0, index).trim();
        this._buffer = this._buffer.subarray(index + 1);
        if (this._buffer.length === 0) this._buffer = undefined;

        if (line.length === 0) return this.readMessage();

        try {
            return JSON.parse(line) as JsonRpcMessage;
        } catch (error) {
            throw new Error(`Failed to parse MCP message: ${line}`);
        }
    }

    clear() {
        this._buffer = undefined;
    }
}

export class StdioMcpTransport implements McpTransport {
    private _process?: ChildProcess;
    private _readBuffer = new MessageBuffer();
    private _serverParams: StdioTransportParams;

    public onclose?: () => void;
    public onerror?: (error: Error) => void;
    public onmessage?: (message: JsonRpcMessage) => void;

    constructor(params: StdioTransportParams) {
        this._serverParams = params;
    }

    async start(): Promise<void> {
        if (this._process) throw new Error("StdioClientTransport already started!");

        return new Promise((resolve, reject) => {
            try {
                this._process = spawn(this._serverParams.command, this._serverParams.args || [], {
                    env: { ...process.env, ...this._serverParams.env },
                    stdio: ["pipe", "pipe", this._serverParams.stderr || "inherit"],
                    shell: false,
                    windowsHide: os.platform() === "win32",
                    cwd: this._serverParams.cwd
                });

                this._process.on("error", (error) => {
                    this.onerror?.(error);
                    reject(error);
                });

                this._process.on("spawn", () => {
                    resolve();
                });

                this._process.on("close", () => {
                    this._process = undefined;
                    this.onclose?.();
                });

                this._process.stdout?.on("data", (data: Buffer) => {
                    this._readBuffer.append(data);
                    this._processReadBuffer();
                });

                this._process.stdout?.on("error", (error) => this.onerror?.(error));
                this._process.stdin?.on("error", (error) => this.onerror?.(error));

            } catch (e: any) {
                reject(e);
            }
        });
    }

    private _processReadBuffer() {
        while (true) {
            try {
                const message = this._readBuffer.readMessage();
                if (!message) break;
                this.onmessage?.(message);
            } catch (error: any) {
                this.onerror?.(error);
            }
        }
    }

    async close(): Promise<void> {
        if (this._process) {
            const proc = this._process;
            this._process = undefined;

            const closePromise = new Promise<void>(resolve => {
                proc.once("close", () => resolve());
            });

            try {
                proc.stdin?.end();
            } catch (e) { }

            const timeout = new Promise<void>(resolve => setTimeout(resolve, 2000));
            const result = await Promise.race([closePromise, timeout]);

            if (proc.exitCode === null) {
                try {
                    proc.kill("SIGTERM");
                } catch (e) { }
                await Promise.race([closePromise, new Promise(resolve => setTimeout(resolve, 2000))]);
            }

            if (proc.exitCode === null) {
                try {
                    proc.kill("SIGKILL");
                } catch (e) { }
            }
        }
        this._readBuffer.clear();
    }

    async send(message: JsonRpcMessage): Promise<void> {
        if (!this._process?.stdin) throw new Error("Not connected");

        const json = JSON.stringify(message) + "\n";

        return new Promise((resolve, reject) => {
            const success = this._process!.stdin!.write(json, (error) => {
                if (error) reject(error);
                else resolve();
            });
            if (!success) {
                this._process!.stdin!.once("drain", resolve);
            }
        });
    }

    get pid() {
        return this._process?.pid;
    }

    get stderr() {
        return this._process?.stderr;
    }
}
