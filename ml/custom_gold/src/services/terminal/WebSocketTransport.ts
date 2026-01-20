
// Logic from chunk_599.ts (WebSocket Transport & Remote Session Management)

import { WebSocket } from "ws";
import { randomUUID } from "node:crypto";

/**
 * Advanced WebSocket transport with connection management and heartbeat.
 */
export class WebSocketTransport {
    private ws: WebSocket | null = null;
    private pingInterval: NodeJS.Timeout | null = null;
    private state: "connecting" | "connected" | "closed" = "connecting";

    constructor(private url: string, private authHeader?: string) { }

    connect() {
        this.ws = new WebSocket(this.url, {
            headers: this.authHeader ? { Authorization: this.authHeader } : {}
        });

        this.ws.on("open", () => {
            this.state = "connected";
            this.startPing();
            console.log("WebSocket connected");
        });

        this.ws.on("close", () => {
            this.state = "closed";
            this.stopPing();
        });
    }

    private startPing() {
        this.pingInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.ping();
            }
        }, 10000);
    }

    private stopPing() {
        if (this.pingInterval) clearInterval(this.pingInterval);
    }

    write(data: any) {
        if (this.state === "connected") {
            this.ws?.send(JSON.stringify(data));
        }
    }

    close() {
        this.ws?.close();
    }
}

// --- Remote Session Utilities ---

/**
 * Detects if a session input is a URL, a local UUID, or a JSONL history file.
 */
export function parseSessionInput(input: string) {
    try {
        const url = new URL(input);
        return { type: "url", value: url.href };
    } catch {
        if (input.match(/^[0-9a-f-]{36}$/)) return { type: "session_id", value: input };
        if (input.endsWith(".jsonl")) return { type: "file", value: input };
    }
    return null;
}

/**
 * Integration for the Claude VS Code extension.
 */
export const VsCodeIntegration = {
    handleEvent: (event: any) => {
        // Dispatch VSCode-specific events to analytics
        console.log("VSCode Event:", event);
    }
};
