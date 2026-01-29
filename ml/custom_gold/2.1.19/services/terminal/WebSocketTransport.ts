import { WebSocket } from "ws";

/**
 * WebSocketTransport class for terminal communication.
 * Deobfuscated and isolated from bundled React logic.
 */
export class WebSocketTransport {
    private ws: WebSocket | null = null;
    private pingInterval: NodeJS.Timeout | null = null;
    private state: "connecting" | "connected" | "closed" = "connecting";
    private messageHandler: ((data: any) => void) | null = null;
    private closeHandler: (() => void) | null = null;

    constructor(private url: string, private options?: { headers?: Record<string, string> }) { }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url, {
                headers: this.options?.headers
            });

            this.ws.on("open", () => {
                this.state = "connected";
                this.startPing();
                resolve();
            });

            this.ws.on("message", (data) => {
                if (this.messageHandler) {
                    this.messageHandler(data.toString());
                }
            });

            this.ws.on("close", () => {
                this.state = "closed";
                this.stopPing();
                if (this.closeHandler) {
                    this.closeHandler();
                }
            });

            this.ws.on("error", (error) => {
                console.error("WebSocket error:", error);
                this.state = "closed";
                this.stopPing();
                reject(error);
            });
        });
    }

    onData(handler: (data: any) => void) {
        this.messageHandler = handler;
    }

    onClose(handler: () => void) {
        this.closeHandler = handler;
    }

    private startPing() {
        this.pingInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.ping();
            } else {
                this.stopPing();
            }
        }, 10000);
    }

    private stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    write(data: any) {
        if (this.state === "connected") {
            try {
                this.ws?.send(data);
            } catch (error) {
                console.error("Error sending WebSocket message:", error);
            }
        }
    }

    send(data: any) {
        this.write(data);
    }

    close() {
        this.ws?.close();
        this.stopPing();
    }
}