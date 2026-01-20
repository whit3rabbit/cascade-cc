
import { JsonRpcMessage } from "./McpSchemas.js";
import { McpTransport } from "./McpTransport.js";

// Custom error for SSE
export class SseError extends Error {
    constructor(public code: number, message: string) {
        super(`SSE error: ${message}`);
        this.name = "SseError";
    }
}

export interface SseTransportOptions {
    requestInit?: RequestInit;
    fetch?: typeof fetch;
    sessionId?: string;
    reconnectionOptions?: {
        initialReconnectionDelay: number;
        maxReconnectionDelay: number;
        reconnectionDelayGrowFactor: number;
        maxRetries: number;
    };
}

export class SseMcpTransport implements McpTransport {
    private _url: URL;
    private _abortController?: AbortController;
    private _sessionId?: string;
    private _reconnectionTimeout?: any;
    private _serverRetryMs?: number;
    private _protocolVersion?: string;
    private _options: SseTransportOptions;

    public onclose?: () => void;
    public onerror?: (error: Error) => void;
    public onmessage?: (message: JsonRpcMessage) => void;

    constructor(url: string | URL, options: SseTransportOptions = {}) {
        this._url = new URL(url);
        this._options = options;
        this._sessionId = options.sessionId;
    }

    async start(): Promise<void> {
        if (this._abortController) throw new Error("Transport already started");
        this._abortController = new AbortController();
        await this._connect();
    }

    private async _connect(resumptionToken?: string) {
        try {
            const headers = new Headers(this._options.requestInit?.headers);
            headers.set("Accept", "text/event-stream");
            if (resumptionToken) headers.set("last-event-id", resumptionToken);
            if (this._sessionId) headers.set("mcp-session-id", this._sessionId);

            const response = await (this._options.fetch ?? fetch)(this._url, {
                ...this._options.requestInit,
                headers,
                signal: this._abortController?.signal
            });

            if (!response.ok) {
                if (response.status === 405) return; // Wait for POST to establish session maybe?
                throw new SseError(response.status, response.statusText);
            }

            this._handleStream(response.body);
        } catch (error: any) {
            this.onerror?.(error);
            this._scheduleReconnection(resumptionToken);
        }
    }

    private _handleStream(body: ReadableStream<Uint8Array> | null) {
        if (!body) return;

        (async () => {
            const reader = body.getReader();
            const decoder = new TextDecoder();
            let currentEventId: string | undefined;

            try {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const text = decoder.decode(value, { stream: true });
                    const lines = text.split("\n");
                    for (const line of lines) {
                        if (line.startsWith("id: ")) {
                            currentEventId = line.substring(4).trim();
                        } else if (line.startsWith("data: ")) {
                            const data = line.substring(6).trim();
                            try {
                                const msg = JSON.parse(data) as JsonRpcMessage;
                                this.onmessage?.(msg);
                            } catch (e) { }
                        } else if (line.startsWith("retry: ")) {
                            this._serverRetryMs = parseInt(line.substring(7).trim(), 10);
                        }
                    }
                }
            } catch (error: any) {
                this.onerror?.(error);
                this._scheduleReconnection(currentEventId);
            }
        })();
    }

    private _scheduleReconnection(token?: string, attempt = 0) {
        if (!this._abortController || this._abortController.signal.aborted) return;

        const maxRetries = this._options.reconnectionOptions?.maxRetries ?? 5;
        if (attempt >= maxRetries) {
            this.onerror?.(new Error("Max reconnection attempts reached"));
            return;
        }

        const delay = this._serverRetryMs ?? 1000 * Math.pow(1.5, attempt);
        this._reconnectionTimeout = setTimeout(() => {
            this._connect(token).catch(err => {
                this._scheduleReconnection(token, attempt + 1);
            });
        }, delay);
    }

    async close(): Promise<void> {
        if (this._reconnectionTimeout) clearTimeout(this._reconnectionTimeout);
        this._abortController?.abort();
        this.onclose?.();
    }

    async send(message: JsonRpcMessage): Promise<void> {
        if (!this._abortController) throw new Error("Not started");

        const headers = new Headers(this._options.requestInit?.headers);
        headers.set("Content-Type", "application/json");
        if (this._sessionId) headers.set("mcp-session-id", this._sessionId);

        const response = await (this._options.fetch ?? fetch)(this._url, {
            ...this._options.requestInit,
            method: "POST",
            headers,
            body: JSON.stringify(message),
            signal: this._abortController.signal
        });

        const newSessionId = response.headers.get("mcp-session-id");
        if (newSessionId) this._sessionId = newSessionId;

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`Failed to send message: ${response.status} ${text}`);
        }
    }

    get sessionId() {
        return this._sessionId;
    }
}
