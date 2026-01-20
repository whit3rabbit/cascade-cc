
import {
    JsonRpcMessage,
    JsonRpcRequest,
    JsonRpcNotification,
    JsonRpcResponse,
    JsonRpcError,
    ErrorCode,
    McpSchemas,
    Task
} from "./McpSchemas.js";
import { McpTransport } from "./McpTransport.js";
import { McpClientCore } from "./McpClientCore.js";

export const RELATED_TASK_META = "io.modelcontextprotocol/related-task";

export class McpError extends Error {
    constructor(public code: number, message: string, public data?: any) {
        super(`MCP error ${code}: ${message}`);
        this.name = "McpError";
    }

    static fromError(code: number, message: string, data?: any): McpError {
        return new McpError(code, message, data);
    }
}

export interface McpClientOptions {
    taskStore?: any;
    taskMessageQueue?: any;
    enforceStrictCapabilities?: boolean;
    debouncedNotificationMethods?: string[];
    defaultTaskPollInterval?: number;
    maxTaskQueueSize?: number;
}

export class McpClient extends McpClientCore {
    private _transport?: McpTransport;
    private _requestMessageId = 0;
    private _requestHandlerAbortControllers = new Map<string | number, AbortController>();
    private _timeoutInfo = new Map<string | number, any>();
    private _pendingDebouncedNotifications = new Set<string>();

    public onclose?: () => void;
    public onerror?: (error: Error) => void;
    public fallbackRequestHandler?: (req: JsonRpcRequest, context: any) => Promise<any>;
    public fallbackNotificationHandler?: (notif: JsonRpcNotification) => Promise<void>;

    constructor(_options: McpClientOptions = {}) {
        super();
        this._options = _options;
        this._taskStore = _options.taskStore;
        this._taskMessageQueue = _options.taskMessageQueue;

        this.setNotificationHandler("notifications/cancelled", async (notif) => {
            const requestId = (notif.params as any)?.requestId;
            const controller = this._requestHandlerAbortControllers.get(requestId);
            if (controller) {
                controller.abort((notif.params as any)?.reason);
            }
        });

        this.setNotificationHandler("notifications/progress", async (notif) => {
            const { progressToken, ...progress } = notif.params as any;
            const handler = this._progressHandlers.get(progressToken);
            if (handler) {
                const timeout = this._timeoutInfo.get(progressToken);
                if (timeout?.resetTimeoutOnProgress) {
                    this._resetTimeout(progressToken);
                }
                handler(progress);
            }
        });
    }

    private _setupTimeout(id: string | number, timeout: number, maxTotalTimeout: number | undefined, onTimeout: () => void, resetOnProgress: boolean = false) {
        this._timeoutInfo.set(id, {
            timeoutId: setTimeout(onTimeout, timeout),
            startTime: Date.now(),
            timeout,
            maxTotalTimeout,
            resetTimeoutOnProgress: resetOnProgress,
            onTimeout
        });
    }

    private _resetTimeout(id: string | number) {
        const info = this._timeoutInfo.get(id);
        if (!info) return false;

        const elapsed = Date.now() - info.startTime;
        if (info.maxTotalTimeout && elapsed >= info.maxTotalTimeout) {
            this._timeoutInfo.delete(id);
            throw new McpError(ErrorCode.RequestTimeout, "Maximum total timeout exceeded", {
                maxTotalTimeout: info.maxTotalTimeout,
                totalElapsed: elapsed
            });
        }

        clearTimeout(info.timeoutId);
        info.timeoutId = setTimeout(info.onTimeout, info.timeout);
        return true;
    }

    private _cleanupTimeout(id: string | number) {
        const info = this._timeoutInfo.get(id);
        if (info) {
            clearTimeout(info.timeoutId);
            this._timeoutInfo.delete(id);
        }
    }

    async connect(transport: McpTransport) {
        this._transport = transport;
        this._transport.onclose = () => this._onclose();
        this._transport.onerror = (err) => this._onerror(err);
        this._transport.onmessage = (msg) => {
            if ("method" in msg) {
                if ("id" in msg) {
                    this._onrequest(msg as JsonRpcRequest);
                } else {
                    this._onnotification(msg as JsonRpcNotification);
                }
            } else if ("id" in msg) {
                this._onresponse(msg as any);
            }
        };
        await this._transport.start();
    }

    private _onclose() {
        const handlers = Array.from(this._responseHandlers.values());
        this._responseHandlers.clear();
        this._progressHandlers.clear();
        this._taskProgressTokens.clear();
        this._pendingDebouncedNotifications.clear();

        const error = new McpError(ErrorCode.ConnectionClosed, "Connection closed");
        this._transport = undefined;
        this.onclose?.();

        for (const handler of handlers) {
            (handler as any)(error);
        }
    }

    protected _onerror(err: Error) {
        this.onerror?.(err);
    }

    private _onnotification(notif: JsonRpcNotification) {
        const handler = this._notificationHandlers.get(notif.method) || this.fallbackNotificationHandler;
        if (handler) {
            Promise.resolve().then(() => handler(notif)).catch(err => {
                this._onerror(new Error(`Uncaught error in notification handler: ${err}`));
            });
        }
    }

    private _onrequest(req: JsonRpcRequest) {
        const handler = this._requestHandlers.get(req.method) || this.fallbackRequestHandler;
        const transport = this._transport;
        const taskId = (req.params as any)?._meta?.[RELATED_TASK_META]?.taskId;

        if (!handler) {
            const errorRes: JsonRpcError = {
                jsonrpc: "2.0",
                id: req.id,
                error: {
                    code: ErrorCode.MethodNotFound,
                    message: "Method not found"
                }
            };
            if (taskId && this._taskMessageQueue) {
                this._enqueueTaskMessage(taskId, { type: "error", message: errorRes, timestamp: Date.now() })
                    .catch(err => this._onerror(new Error(`Failed to enqueue error: ${err}`)));
            } else {
                transport?.send(errorRes).catch(err => this._onerror(new Error(`Failed to send error: ${err}`)));
            }
            return;
        }

        const controller = new AbortController();
        this._requestHandlerAbortControllers.set(req.id, controller);

        const context = {
            signal: controller.signal,
            sessionId: (transport as any)?.sessionId,
            taskId,
            sendNotification: async (n: JsonRpcNotification) => {
                await this.notification(n.method, n.params, { relatedRequestId: req.id, relatedTask: taskId ? { taskId } : undefined });
            },
            sendRequest: async (method: string, params?: any, opts: any = {}) => {
                return await this.request({ method, params }, null, { ...opts, relatedRequestId: req.id, relatedTask: taskId ? { taskId } : opts.relatedTask });
            },
            taskStore: this._taskStore ? this.requestTaskStore(req, (transport as any)?.sessionId) : undefined
        };

        Promise.resolve().then(() => handler(req, context)).then(async (result) => {
            if (controller.signal.aborted) return;
            const res: JsonRpcResponse = { jsonrpc: "2.0", id: req.id, result };
            if (taskId && this._taskMessageQueue) {
                await this._enqueueTaskMessage(taskId, { type: "response", message: res, timestamp: Date.now() }, (transport as any)?.sessionId);
            } else {
                await transport?.send(res);
            }
        }, async (error) => {
            if (controller.signal.aborted) return;
            const errRes: JsonRpcError = {
                jsonrpc: "2.0",
                id: req.id,
                error: {
                    code: error.code || ErrorCode.InternalError,
                    message: error.message || "Internal error",
                    data: error.data
                }
            };
            if (taskId && this._taskMessageQueue) {
                await this._enqueueTaskMessage(taskId, { type: "error", message: errRes, timestamp: Date.now() }, (transport as any)?.sessionId);
            } else {
                await transport?.send(errRes);
            }
        }).catch(err => {
            this._onerror(new Error(`Failed to send response: ${err}`));
        }).finally(() => {
            this._requestHandlerAbortControllers.delete(req.id);
        });
    }

    private _onresponse(res: JsonRpcResponse | JsonRpcError) {
        const id = res.id;
        const resolver = this._requestResolvers.get(id);
        if (resolver) {
            this._requestResolvers.delete(id);
            if ("error" in res) {
                resolver(new McpError(res.error.code, res.error.message, res.error.data));
            } else {
                resolver(res as JsonRpcResponse);
            }
            return;
        }

        const handler = this._responseHandlers.get(id);
        if (!handler) {
            this._onerror(new Error(`Received response for unknown message ID: ${id}`));
            return;
        }

        this._responseHandlers.delete(id);
        this._cleanupTimeout(id);

        let isTask = false;
        if ("result" in res && (res.result as any)?.task?.taskId) {
            isTask = true;
            this._taskProgressTokens.set((res.result as any).task.taskId, id);
        }

        if (!isTask) {
            this._progressHandlers.delete(id);
        }

        if ("error" in res) {
            handler(new McpError(res.error.code, res.error.message, res.error.data));
        } else {
            handler(res as JsonRpcResponse);
        }
    }

    async request(req: { method: string, params?: any }, schema: any, options: any = {}): Promise<any> {
        if (!this._transport) throw new Error("Not connected");

        return new Promise((resolve, reject) => {
            const id = this._requestMessageId++;
            const jsonRpcReq: JsonRpcRequest = {
                jsonrpc: "2.0",
                id,
                method: req.method,
                params: {
                    ...req.params,
                    ...(options.onprogress ? { _meta: { progressToken: id } } : {})
                }
            };

            if (options.relatedTask) {
                jsonRpcReq.params = {
                    ...jsonRpcReq.params,
                    _meta: {
                        ...(jsonRpcReq.params as any)?._meta,
                        [RELATED_TASK_META]: options.relatedTask
                    }
                };
            }

            if (options.onprogress) {
                this._progressHandlers.set(id, options.onprogress);
            }

            const timeout = options.timeout || 60000;
            const onTimeout = () => {
                reject(new McpError(ErrorCode.RequestTimeout, "Request timed out"));
            };

            this._setupTimeout(id, timeout, options.maxTotalTimeout, onTimeout, options.resetTimeoutOnProgress);

            this._responseHandlers.set(id, (res: any) => {
                if (res instanceof McpError) {
                    reject(res);
                } else {
                    if (schema) {
                        try {
                            const validated = schema.parse(res.result);
                            resolve(validated);
                        } catch (e) {
                            reject(new McpError(ErrorCode.InternalError, `Response validation failed: ${e}`));
                        }
                    } else {
                        resolve(res.result);
                    }
                }
            });

            this._transport!.send(jsonRpcReq, options).catch(err => {
                this._cleanupTimeout(id);
                this._responseHandlers.delete(id);
                reject(err);
            });
        });
    }

    async notification(method: string, params?: any, options: any = {}) {
        if (!this._transport) throw new Error("Not connected");

        const notif: JsonRpcNotification = {
            jsonrpc: "2.0",
            method,
            params
        };

        if (options.relatedTask) {
            notif.params = {
                ...notif.params,
                _meta: {
                    ...notif.params?._meta,
                    [RELATED_TASK_META]: options.relatedTask
                }
            };
        }

        await this._transport.send(notif, options);
    }

    async close() {
        await this._transport?.close();
        this._onclose();
    }
}
