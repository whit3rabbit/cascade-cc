
import { Readable, Writable } from "stream";
import { Message, ResponseMessage, RequestMessage, NotificationMessage, ResponseError, ErrorCodes } from "./types.js";

// Simplified JSON-RPC connection implementation replacing vscode-jsonrpc
export class StreamMessageReader {
    private readable: Readable;
    private buffer: Buffer = Buffer.alloc(0);
    private contentLength: number = -1;
    private onMessageCallback?: (message: Message) => void;
    private onErrorCallback?: (error: Error) => void;
    private onCloseCallback?: () => void;

    constructor(readable: Readable) {
        this.readable = readable;
        this.readable.on("data", this.onData.bind(this));
        this.readable.on("error", (err) => this.onErrorCallback?.(err));
        this.readable.on("end", () => this.onCloseCallback?.());
    }

    private onData(data: Buffer) {
        this.buffer = Buffer.concat([this.buffer, data]);
        this.processBuffer();
    }

    private processBuffer() {
        while (true) {
            if (this.contentLength === -1) {
                const headerEnd = this.buffer.indexOf("\r\n\r\n");
                if (headerEnd === -1) return;

                const headers = this.buffer.slice(0, headerEnd).toString();
                const contentLengthMatch = headers.match(/Content-Length: (\d+)/i);
                if (!contentLengthMatch) {
                    // Invalid header or missing Content-Length?
                    // Just skip header? Or erro?
                    // For simplicity assuming correct header
                    return;
                }

                this.contentLength = parseInt(contentLengthMatch[1], 10);
                this.buffer = this.buffer.slice(headerEnd + 4);
            }

            if (this.buffer.length >= this.contentLength) {
                const content = this.buffer.slice(0, this.contentLength).toString("utf8");
                this.buffer = this.buffer.slice(this.contentLength);
                this.contentLength = -1;

                try {
                    const message = JSON.parse(content);
                    this.onMessageCallback?.(message);
                } catch (e) {
                    this.onErrorCallback?.(new Error("Failed to parse JSON-RPC message"));
                }
            } else {
                return;
            }
        }
    }

    public listen(callback: (message: Message) => void) {
        this.onMessageCallback = callback;
    }

    public onError(callback: (error: Error) => void) {
        this.onErrorCallback = callback;
    }

    public onClose(callback: () => void) {
        this.onCloseCallback = callback;
    }
}

export class StreamMessageWriter {
    private writable: Writable;
    private errorCallback?: (error: any) => void;

    constructor(writable: Writable) {
        this.writable = writable;
        this.writable.on("error", (err) => this.errorCallback?.(err));
    }

    public async write(message: Message): Promise<void> {
        const content = JSON.stringify(message);
        const length = Buffer.byteLength(content, "utf8");
        const headers = `Content-Length: ${length}\r\n\r\n`;
        return new Promise((resolve, reject) => {
            const canWrite = this.writable.write(headers + content, "utf8", (error) => {
                if (error) {
                    reject(error);
                } else {
                    resolve();
                }
            });
            if (!canWrite) {
                // backpressure ?
            }
        });
    }

    public onError(callback: (error: any) => void) {
        this.errorCallback = callback;
    }
}

export interface MessageConnection {
    sendRequest(method: string, params: any): Promise<any>;
    sendNotification(method: string, params: any): void;
    onNotification(method: string, handler: (params: any) => void): void;
    onRequest(method: string, handler: (params: any) => Promise<any>): void;
    listen(): void;
    dispose(): void;
    onError(handler: (error: any) => void): void;
    onClose(handler: () => void): void;
    trace(value: any, tracer: any): Promise<void>;
}

export function createMessageConnection(reader: StreamMessageReader, writer: StreamMessageWriter): MessageConnection {
    const requestHandlers = new Map<string, (params: any) => Promise<any>>();
    const notificationHandlers = new Map<string, (params: any) => void>();
    const pendingRequests = new Map<number | string, { resolve: (res: any) => void, reject: (err: any) => void }>();

    let nextId = 0;
    let isDisposed = false;

    reader.listen((message: Message) => {
        if (isDisposed) return;

        if (isRequestMessage(message)) {
            const handler = requestHandlers.get(message.method);
            if (handler) {
                handler(message.params).then(result => {
                    writer.write({
                        jsonrpc: "2.0",
                        id: message.id,
                        result
                    } as ResponseMessage);
                }, error => {
                    writer.write({
                        jsonrpc: "2.0",
                        id: message.id,
                        error: error instanceof ResponseError ? error : new ResponseError(ErrorCodes.InternalError, String(error))
                    } as ResponseMessage);
                });
            } else {
                // Method not found
            }
        } else if (isNotificationMessage(message)) {
            const handler = notificationHandlers.get(message.method);
            if (handler) {
                handler(message.params);
            }
        } else if (isResponseMessage(message)) {
            if (message.id !== null && pendingRequests.has(message.id)) {
                const { resolve, reject } = pendingRequests.get(message.id)!;
                pendingRequests.delete(message.id);
                if (message.error) {
                    reject(new ResponseError(message.error.code, message.error.message, message.error.data));
                } else {
                    resolve(message.result);
                }
            }
        }
    });

    return {
        sendRequest: async (method: string, params: any) => {
            const id = nextId++;
            return new Promise((resolve, reject) => {
                pendingRequests.set(id, { resolve, reject });
                writer.write({
                    jsonrpc: "2.0",
                    id,
                    method,
                    params
                } as RequestMessage).catch(reject);
            });
        },
        sendNotification: (method: string, params: any) => {
            writer.write({
                jsonrpc: "2.0",
                method,
                params
            } as NotificationMessage);
        },
        onNotification: (method: string, handler: (params: any) => void) => {
            notificationHandlers.set(method, handler);
        },
        onRequest: (method: string, handler: (params: any) => Promise<any>) => {
            requestHandlers.set(method, handler);
        },
        listen: () => {
            // Already listening in constructor/setup
        },
        dispose: () => {
            isDisposed = true;
            pendingRequests.forEach(p => p.reject(new Error("Connection disposed")));
            pendingRequests.clear();
        },
        onError: (handler) => {
            reader.onError(handler);
            writer.onError(handler);
        },
        onClose: (handler) => {
            reader.onClose(handler);
        },
        trace: async () => { }
    };
}

function isRequestMessage(message: Message): message is RequestMessage {
    return (message as any).method && (message as any).id !== undefined;
}

function isNotificationMessage(message: Message): message is NotificationMessage {
    return (message as any).method && (message as any).id === undefined;
}

function isResponseMessage(message: Message): message is ResponseMessage {
    return (message as any).result !== undefined || (message as any).error !== undefined;
}
