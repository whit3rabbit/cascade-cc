
import {
    JsonRpcRequest,
    JsonRpcNotification,
    JsonRpcResponse,
    ErrorCode,
    McpSchemas,
    Task
} from "./McpSchemas.js";
import { RELATED_TASK_META, McpError } from "./McpClient.js";

export abstract class McpClientCore {
    protected _requestHandlers = new Map<string, (req: JsonRpcRequest, context: any) => Promise<any>>();
    protected _notificationHandlers = new Map<string, (notif: JsonRpcNotification) => Promise<void>>();
    protected _taskStore: any;
    protected _taskMessageQueue: any;
    protected _options: any;
    protected _taskProgressTokens = new Map<string, string | number>();
    protected _requestResolvers = new Map<string | number, (res: JsonRpcResponse | McpError) => void>();
    protected _progressHandlers = new Map<string | number, (progress: any) => void>();
    protected _responseHandlers = new Map<string | number, (res: JsonRpcResponse | McpError) => void>();

    abstract notification(method: string, params?: any, options?: any): Promise<void>;
    abstract request(req: { method: string, params?: any }, schema: any, options?: any): Promise<any>;
    protected abstract _onerror(err: Error): void;

    setRequestHandler(method: string, handler: (req: JsonRpcRequest, context: any) => Promise<any>) {
        this._requestHandlers.set(method, handler);
    }

    removeRequestHandler(method: string) {
        this._requestHandlers.delete(method);
    }

    setNotificationHandler(method: string, handler: (notif: JsonRpcNotification) => Promise<void>) {
        this._notificationHandlers.set(method, handler);
    }

    removeNotificationHandler(method: string) {
        this._notificationHandlers.delete(method);
    }

    protected _cleanupTaskProgressHandler(taskId: string) {
        const token = this._taskProgressTokens.get(taskId);
        if (token !== undefined) {
            this._progressHandlers.delete(token);
            this._taskProgressTokens.delete(taskId);
        }
    }

    protected async _enqueueTaskMessage(taskId: string, message: any, sessionId?: string) {
        if (!this._taskStore || !this._taskMessageQueue) {
            throw new Error("Cannot enqueue task message: taskStore and taskMessageQueue are not configured");
        }
        const maxSize = this._options?.maxTaskQueueSize;
        await this._taskMessageQueue.enqueue(taskId, message, sessionId, maxSize);
    }

    protected async _clearTaskQueue(taskId: string, sessionId?: string) {
        if (this._taskMessageQueue) {
            const messages = await this._taskMessageQueue.dequeueAll(taskId, sessionId);
            for (const msg of messages) {
                if (msg.type === "request") {
                    const id = msg.message.id;
                    const resolver = this._requestResolvers.get(id);
                    if (resolver) {
                        resolver(new McpError(ErrorCode.InternalError, "Task cancelled or completed"));
                        this._requestResolvers.delete(id);
                    } else {
                        this._onerror(new Error(`Resolver missing for request ${id} during task ${taskId} cleanup`));
                    }
                }
            }
        }
    }

    protected async _waitForTaskUpdate(taskId: string, signal: AbortSignal): Promise<void> {
        let interval = this._options?.defaultTaskPollInterval ?? 1000;
        try {
            const task = await this._taskStore?.getTask(taskId);
            if (task?.pollInterval) {
                interval = task.pollInterval;
            }
        } catch (e) { }

        return new Promise((resolve, reject) => {
            if (signal.aborted) {
                reject(new McpError(ErrorCode.InvalidRequest, "Request cancelled"));
                return;
            }
            const timer = setTimeout(resolve, interval);
            signal.addEventListener("abort", () => {
                clearTimeout(timer);
                reject(new McpError(ErrorCode.InvalidRequest, "Request cancelled"));
            }, { once: true });
        });
    }

    protected requestTaskStore(req: JsonRpcRequest, sessionId?: string) {
        const store = this._taskStore;
        if (!store) throw new Error("No task store configured");

        return {
            createTask: async (params: any) => {
                return await store.createTask(params, req.id, { method: req.method, params: req.params }, sessionId);
            },
            getTask: async (taskId: string) => {
                const task = await store.getTask(taskId, sessionId);
                if (!task) throw new McpError(ErrorCode.InvalidParams, "Failed to retrieve task: Task not found");
                return task;
            },
            storeTaskResult: async (taskId: string, result: any, meta?: any) => {
                await store.storeTaskResult(taskId, result, meta, sessionId);
                const task = await store.getTask(taskId, sessionId);
                if (task) {
                    await this.notification("notifications/tasks/status", task);
                    if (this._isTerminalStatus(task.status)) {
                        this._cleanupTaskProgressHandler(taskId);
                    }
                }
            },
            getTaskResult: (taskId: string) => {
                return store.getTaskResult(taskId, sessionId);
            },
            updateTaskStatus: async (taskId: string, status: string, message?: string) => {
                const task = await store.getTask(taskId, sessionId);
                if (!task) throw new McpError(ErrorCode.InvalidParams, `Task "${taskId}" not found`);
                if (this._isTerminalStatus(task.status)) {
                    throw new McpError(ErrorCode.InvalidParams, `Cannot update task "${taskId}" from terminal status "${task.status}"`);
                }
                await store.updateTaskStatus(taskId, status, message, sessionId);
                const updated = await store.getTask(taskId, sessionId);
                if (updated) {
                    await this.notification("notifications/tasks/status", updated);
                    if (this._isTerminalStatus(updated.status)) {
                        this._cleanupTaskProgressHandler(taskId);
                    }
                }
            },
            listTasks: (cursor?: string) => {
                return store.listTasks(cursor, sessionId);
            }
        };
    }

    protected _isTerminalStatus(status: string) {
        return status === "completed" || status === "failed" || status === "cancelled";
    }
}
