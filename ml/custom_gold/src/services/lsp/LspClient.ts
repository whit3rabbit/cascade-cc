
import { spawn, ChildProcess, SpawnOptions } from "node:child_process";
import { MessageConnection, createMessageConnection, StreamMessageReader, StreamMessageWriter } from "./LspConnection.js";
import { log } from "../logger/loggerService.js";

const logger = log("lsp-client");

export interface LspClientOptions {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
}

/**
 * Handles communication with an LSP server process.
 * Based on chunk_371.ts (R52)
 */
export class LspClient {
    private connection?: MessageConnection;
    private process?: ChildProcess;
    private _capabilities: any;
    private _isInitialized = false;
    private hasFailedToStart = false;
    private startError?: Error;
    private isStopping = false;

    private pendingNotifications: { method: string, handler: (params: any) => void }[] = [];
    private pendingRequests: { method: string, handler: (params: any) => Promise<any> }[] = [];

    constructor(public readonly name: string) { }

    get capabilities() {
        return this._capabilities;
    }

    get isInitialized() {
        return this._isInitialized;
    }

    private checkError() {
        if (this.hasFailedToStart) {
            throw this.startError || new Error(`LSP server ${this.name} failed to start`);
        }
    }

    async start(command: string, args: string[], options?: LspClientOptions) {
        try {
            const spawnOptions: SpawnOptions = {
                stdio: ["pipe", "pipe", "pipe"],
                env: options?.env ? { ...process.env, ...options.env } : undefined,
                cwd: options?.cwd
            };

            this.process = spawn(command, args, spawnOptions);

            if (!this.process.stdout || !this.process.stdin) {
                throw new Error("LSP server process stdio not available");
            }

            const proc = this.process;
            await new Promise<void>((resolve, reject) => {
                const onSpawn = () => {
                    cleanup();
                    resolve();
                };
                const onError = (err: Error) => {
                    cleanup();
                    reject(err);
                };
                const cleanup = () => {
                    proc.removeListener("spawn", onSpawn);
                    proc.removeListener("error", onError);
                };
                proc.once("spawn", onSpawn);
                proc.once("error", onError);
            });

            if (this.process.stderr) {
                this.process.stderr.on("data", (data) => {
                    const msg = data.toString().trim();
                    if (msg) {
                        logger.info(`[LSP SERVER ${this.name}] ${msg}`);
                    }
                });
            }

            this.process.on("error", (err) => {
                if (!this.isStopping) {
                    this.hasFailedToStart = true;
                    this.startError = err;
                    logger.error(new Error(`LSP server ${this.name} failed to start: ${err.message}`));
                }
            });

            this.process.on("exit", (code, signal) => {
                if (!this.isStopping) {
                    this._isInitialized = false;
                    this.hasFailedToStart = false;
                    this.startError = undefined;
                    if (code !== 0 && code !== null) {
                        logger.error(new Error(`LSP server ${this.name} crashed with exit code ${code}`));
                    } else if (signal) {
                        logger.error(new Error(`LSP server ${this.name} killed by signal ${signal}`));
                    }
                }
            });

            this.process.stdin.on("error", (err) => {
                if (!this.isStopping) {
                    logger.error(`LSP server ${this.name} stdin error: ${err.message}`);
                }
            });

            const reader = new StreamMessageReader(this.process.stdout);
            const writer = new StreamMessageWriter(this.process.stdin);
            this.connection = createMessageConnection(reader, writer);

            this.connection.onError((error) => {
                if (!this.isStopping) {
                    this.hasFailedToStart = true;
                    this.startError = error;
                    logger.error(new Error(`LSP server ${this.name} connection error: ${error.message}`));
                }
            });

            this.connection.onClose(() => {
                if (!this.isStopping) {
                    this._isInitialized = false;
                    logger.info(`LSP server ${this.name} connection closed`);
                }
            });

            this.connection.listen();

            // apply queued handlers
            for (const { method, handler } of this.pendingNotifications) {
                this.connection.onNotification(method, handler);
            }
            this.pendingNotifications = [];

            for (const { method, handler } of this.pendingRequests) {
                this.connection.onRequest(method, handler);
            }
            this.pendingRequests = [];

            logger.info(`LSP client started for ${this.name}`);

        } catch (err: any) {
            const error = new Error(`LSP server ${this.name} failed to start: ${err.message}`);
            logger.error(error);
            throw err;
        }
    }

    async initialize(params: any): Promise<any> {
        if (!this.connection) throw new Error("LSP client not started");
        this.checkError();
        try {
            const result = await this.connection.sendRequest("initialize", params);
            this._capabilities = result.capabilities;
            await this.connection.sendNotification("initialized", {});
            this._isInitialized = true;
            logger.info(`LSP server ${this.name} initialized`);
            return result;
        } catch (err: any) {
            logger.error(new Error(`LSP server ${this.name} initialize failed: ${err.message}`));
            throw err;
        }
    }

    async sendRequest(method: string, params: any) {
        if (!this.connection) throw new Error("LSP client not started");
        this.checkError();
        if (!this._isInitialized && method !== "initialize") {
            throw new Error("LSP server not initialized");
        }
        try {
            return await this.connection.sendRequest(method, params);
        } catch (err: any) {
            logger.error(new Error(`LSP server ${this.name} request ${method} failed: ${err.message}`));
            throw err;
        }
    }

    async sendNotification(method: string, params: any) {
        if (!this.connection) throw new Error("LSP client not started");
        this.checkError();
        try {
            await this.connection.sendNotification(method, params);
        } catch (err: any) {
            logger.error(new Error(`LSP server ${this.name} notification ${method} failed: ${err.message}`));
        }
    }

    onNotification(method: string, handler: (params: any) => void) {
        if (!this.connection) {
            this.pendingNotifications.push({ method, handler });
            return;
        }
        this.checkError();
        this.connection.onNotification(method, handler);
    }

    onRequest(method: string, handler: (params: any) => Promise<any>) {
        if (!this.connection) {
            this.pendingRequests.push({ method, handler });
            return;
        }
        this.checkError();
        this.connection.onRequest(method, handler);
    }

    async stop() {
        this.isStopping = true;
        let lastError: any;
        try {
            if (this.connection) {
                try {
                    await this.connection.sendRequest("shutdown", null);
                    await this.connection.sendNotification("exit", null);
                } catch (e) {
                    lastError = e;
                    logger.error(new Error(`LSP server ${this.name} stop failed: ${e instanceof Error ? e.message : String(e)}`));
                }
            }
        } finally {
            if (this.connection) {
                try {
                    this.connection.dispose();
                } catch (e: any) {
                    logger.error(`Connection disposal failed for ${this.name}: ${e.message}`);
                }
                this.connection = undefined;
            }
            if (this.process) {
                this.process.removeAllListeners("error");
                this.process.removeAllListeners("exit");
                if (this.process.stdin) {
                    this.process.stdin.removeAllListeners("error");
                }
                if (this.process.stderr) {
                    this.process.stderr.removeAllListeners("data");
                }
                try {
                    this.process.kill();
                } catch (e: any) {
                    logger.error(`Process kill failed for ${this.name}: ${e.message}`);
                }
                this.process = undefined;
            }
            this._isInitialized = false;
            this._capabilities = undefined;
            this.isStopping = false;
            if (lastError) {
                this.hasFailedToStart = true;
                this.startError = lastError;
            }
            logger.info(`LSP client stopped for ${this.name}`);
        }
        if (lastError) throw lastError;
    }
}
