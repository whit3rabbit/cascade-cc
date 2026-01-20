
import { WebSocket } from "ws";
import { JsonRpcMessage } from "./McpSchemas.js";
import { McpTransport } from "./McpTransport.js";

export class WebSocketMcpTransport implements McpTransport {
    private _started = false;
    private _opened: Promise<void>;

    public onclose?: () => void;
    public onerror?: (error: Error) => void;
    public onmessage?: (message: JsonRpcMessage) => void;

    constructor(private _ws: WebSocket) {
        this._opened = new Promise((resolve, reject) => {
            if (this._ws.readyState === WebSocket.OPEN) {
                resolve();
            } else {
                this._ws.once("open", () => resolve());
                this._ws.once("error", (err) => reject(err));
            }
        });

        this._ws.on("message", this._onMessageHandler);
        this._ws.on("error", this._onErrorHandler);
        this._ws.on("close", this._onCloseHandler);
    }

    private _onMessageHandler = (data: any) => {
        try {
            const message = JSON.parse(data.toString("utf-8")) as JsonRpcMessage;
            this.onmessage?.(message);
        } catch (error) {
            this._onErrorHandler(error instanceof Error ? error : new Error("Failed to parse message"));
        }
    };

    private _onErrorHandler = (error: Error) => {
        this.onerror?.(error);
    };

    private _onCloseHandler = () => {
        this.onclose?.();
        this._ws.off("message", this._onMessageHandler);
        this._ws.off("error", this._onErrorHandler);
        this._ws.off("close", this._onCloseHandler);
    };

    async start(): Promise<void> {
        if (this._started) throw new Error("Transport already started");
        await this._opened;
        if (this._ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket is not open");
        }
        this._started = true;
    }

    async close(): Promise<void> {
        if (this._ws.readyState === WebSocket.OPEN || this._ws.readyState === WebSocket.CONNECTING) {
            this._ws.close();
        }
        this._onCloseHandler();
    }

    async send(message: JsonRpcMessage): Promise<void> {
        if (this._ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket is not open");
        }
        const json = JSON.stringify(message);
        return new Promise((resolve, reject) => {
            this._ws.send(json, (err) => {
                if (err) reject(err);
                else resolve();
            });
        });
    }
}
