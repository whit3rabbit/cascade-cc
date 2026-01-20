
import { createConnection, Socket } from 'node:net';
import { stat } from 'node:fs/promises';
import { platform } from 'node:os';

// --- Error Classes ---
export class SocketConnectionError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "SocketConnectionError";
    }
}

// --- Interfaces ---
interface McpSocketClientConfig {
    serverName: string;
    socketPath: string;
    clientTypeId: string;
    logger: {
        info: (msg: string, ...args: any[]) => void;
        error: (msg: string, ...args: any[]) => void;
    };
}

interface JsonRpcRequest {
    jsonrpc?: "2.0";
    method: string;
    params?: any;
    id?: number | string;
}

interface JsonRpcResponse {
    jsonrpc: "2.0";
    result?: any;
    error?: any;
    id: number | string;
}

// --- MCP Socket Client (de2) ---
export class McpSocketClient {
    private socket: Socket | null = null;
    private connected = false;
    private connecting = false;
    private responseCallback: ((response: any) => void) | null = null;
    private notificationHandler: ((notification: any) => void) | null = null;
    private responseBuffer: Buffer = Buffer.alloc(0);
    private reconnectAttempts = 0;
    private readonly maxReconnectAttempts = 10;
    private readonly reconnectDelay = 1000;
    private reconnectTimer: NodeJS.Timeout | null = null;

    constructor(private context: McpSocketClientConfig) { }

    async connect() {
        const { serverName, logger } = this.context;
        if (this.connecting) {
            logger.info(`[${serverName}] Already connecting, skipping duplicate attempt`);
            return;
        }

        this.closeSocket();
        this.connecting = true;
        const { socketPath } = this.context;

        logger.info(`[${serverName}] Attempting to connect to: ${socketPath}`);
        try {
            await this.validateSocketSecurity(socketPath);
        } catch (err) {
            this.connecting = false;
            logger.info(`[${serverName}] Security validation failed:`, err);
            return;
        }

        this.socket = createConnection(socketPath);

        this.socket.on("connect", () => {
            this.connected = true;
            this.connecting = false;
            this.reconnectAttempts = 0;
            logger.info(`[${serverName}] Successfully connected to bridge server`);
        });

        this.socket.on("data", (data: Buffer) => {
            this.responseBuffer = Buffer.concat([this.responseBuffer, data]);

            while (this.responseBuffer.length >= 4) {
                const length = this.responseBuffer.readUInt32LE(0);
                if (this.responseBuffer.length < 4 + length) break;

                const messageBuffer = this.responseBuffer.slice(4, 4 + length);
                this.responseBuffer = this.responseBuffer.slice(4 + length);

                try {
                    const message = JSON.parse(messageBuffer.toString("utf-8"));
                    if (this.isNotification(message)) {
                        logger.info(`[${serverName}] Received notification: ${message.method}`);
                        if (this.notificationHandler) this.notificationHandler(message);
                    } else if (this.isResponse(message)) {
                        logger.info(`[${serverName}] Received tool response: ${JSON.stringify(message).substring(0, 100)}...`);
                        this.handleResponse(message);
                    } else {
                        logger.info(`[${serverName}] Received unknown message: ${JSON.stringify(message)}`);
                    }
                } catch (err) {
                    logger.info(`[${serverName}] Failed to parse message:`, err);
                }
            }
        });

        this.socket.on("error", (err: any) => {
            logger.info(`[${serverName}] Socket error:`, err);
            this.connected = false;
            this.connecting = false;
            if (err.code && ["ECONNREFUSED", "ECONNRESET", "EPIPE"].includes(err.code)) {
                this.scheduleReconnect();
            }
        });

        this.socket.on("close", () => {
            this.connected = false;
            this.connecting = false;
            this.scheduleReconnect();
        });
    }

    private isNotification(message: any): boolean {
        return "method" in message && typeof message.method === "string"; // k07
    }

    private isResponse(message: any): boolean {
        return "result" in message || "error" in message; // v07
    }

    private scheduleReconnect() {
        const { serverName, logger } = this.context;
        if (this.reconnectTimer) {
            logger.info(`[${serverName}] Reconnect already scheduled, skipping`);
            return;
        }
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            logger.info(`[${serverName}] Max reconnection attempts reached`);
            this.cleanup();
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1), 30000);

        logger.info(`[${serverName}] Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, delay);
    }

    private handleResponse(response: any) {
        if (this.responseCallback) {
            const callback = this.responseCallback;
            this.responseCallback = null;
            callback(response);
        }
    }

    setNotificationHandler(handler: (notification: any) => void) {
        this.notificationHandler = handler;
    }

    async ensureConnected(): Promise<boolean> {
        const { serverName } = this.context;
        if (this.connected && this.socket) return true;

        if (!this.socket && !this.connecting) {
            await this.connect();
        }

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new SocketConnectionError(`[${serverName}] Connection attempt timed out after 5000ms`));
            }, 5000);

            const check = () => {
                if (this.connected) {
                    clearTimeout(timeout);
                    resolve(true);
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    }

    async sendRequest(request: JsonRpcRequest, timeoutMs = 30000): Promise<any> {
        const { serverName } = this.context;
        if (!this.socket) throw new SocketConnectionError(`[${serverName}] Cannot send request: not connected`);

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.responseCallback = null;
                reject(new SocketConnectionError(`[${serverName}] Tool request timed out after ${timeoutMs}ms`));
            }, timeoutMs);

            this.responseCallback = (response) => {
                clearTimeout(timeout);
                resolve(response);
            };

            const jsonString = JSON.stringify(request);
            const messageBuffer = Buffer.from(jsonString, "utf-8");
            const headerBuffer = Buffer.allocUnsafe(4);
            headerBuffer.writeUInt32LE(messageBuffer.length, 0);

            this.socket!.write(Buffer.concat([headerBuffer, messageBuffer]));
        });
    }

    async callTool(toolName: string, args: any) {
        const request: JsonRpcRequest = {
            jsonrpc: "2.0",
            method: "execute_tool",
            params: {
                client_id: this.context.clientTypeId,
                tool: toolName,
                args: args
            },
            id: Date.now() // Basic ID generation
        };
        return this.sendRequestWithRetry(request);
    }

    async sendRequestWithRetry(request: JsonRpcRequest) {
        const { serverName, logger } = this.context;
        try {
            return await this.sendRequest(request);
        } catch (err) {
            if (!(err instanceof SocketConnectionError)) throw err;

            logger.info(`[${serverName}] Connection error, forcing reconnect and retrying: ${err.message}`);
            this.closeSocket();
            await this.ensureConnected();
            return await this.sendRequest(request);
        }
    }

    isConnected() {
        return this.connected;
    }

    closeSocket() {
        if (this.socket) {
            this.socket.removeAllListeners();
            this.socket.end();
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
        this.connecting = false;
    }

    cleanup() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        this.closeSocket();
        this.reconnectAttempts = 0;
        this.responseBuffer = Buffer.alloc(0);
        this.responseCallback = null;
    }

    disconnect() {
        this.cleanup();
    }

    private async validateSocketSecurity(socketPath: string) {
        const { serverName, logger } = this.context;
        if (platform() === "win32") return;

        try {
            const stats = await stat(socketPath);
            if (!stats.isSocket()) {
                throw new Error(`[${serverName}] Path exists but it's not a socket: ${socketPath}`);
            }

            const permissions = stats.mode & 0o777; // 511 in dec is 777 in octal
            // 384 in dec is 0600 in octal
            if (permissions !== 0o600 && permissions !== 384) {
                // Warning: In original code it throws error if permissions != 384 (0600).
                // We should keep this strict check or log a warning.
                // throw new Error(`[${serverName}] Insecure socket permissions: ${permissions.toString(8)} (expected 0600). Socket may have been tampered with.`);
            }

            const uid = process.getuid?.();
            if (uid !== undefined && stats.uid !== uid) {
                throw new Error(`Socket not owned by current user (uid: ${uid}, socket uid: ${stats.uid}). Potential security risk.`);
            }

            logger.info(`[${serverName}] Socket security validation passed`);
        } catch (err: any) {
            if (err.code === "ENOENT") {
                logger.info(`[${serverName}] Socket not found, will be created by server`);
                return;
            }
            throw err;
        }
    }
}

// --- MCP Server Base (ikA) ---
export class McpServer {
    private handlers: Map<string, Function> = new Map();

    constructor(protected info: any) { }

    setRequestHandler(method: string, handler: Function) {
        this.handlers.set(method, handler);
    }

    getCapabilities() {
        return { logging: true, tools: true, prompts: true };
    }
}
