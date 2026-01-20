
import { randomUUID } from "node:crypto";
import { log } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { createUserMessage } from "../terminal/MessageFactory.js";
import { executeToolWithHooks } from "../hooks/HookExecutor.js";

const logger = log("loops");

export interface ToolExecutionResult {
    message?: any;
    contextModifier?: (context: any) => any;
}

export type ToolStatus = "queued" | "executing" | "completed" | "yielded";

export interface ToolState {
    id: string;
    block: any;
    assistantMessage: any;
    status: ToolStatus;
    isConcurrencySafe: boolean;
    pendingProgress: any[];
    results?: any[];
    contextModifiers?: any[];
    promise?: Promise<void>;
}

/**
 * Manages concurrent and sequential tool execution with hooks.
 * Based on chunk_456.ts (_F0)
 */
export class ToolExecutionQueue {
    private toolDefinitions: any[];
    private canUseTool: any;
    private tools: ToolState[] = [];
    private toolUseContext: any;
    private hasErrored = false;
    private progressAvailableResolve?: () => void;

    constructor(toolDefinitions: any[], canUseTool: any, toolUseContext: any) {
        this.toolDefinitions = toolDefinitions;
        this.canUseTool = canUseTool;
        this.toolUseContext = toolUseContext;
    }

    addTool(toolUse: any, assistantMessage: any) {
        const definition = this.toolDefinitions.find(t => t.name === toolUse.name);
        if (!definition) {
            this.tools.push({
                id: toolUse.id,
                block: toolUse,
                assistantMessage,
                status: "completed",
                isConcurrencySafe: true,
                pendingProgress: [],
                results: [createUserMessage([
                    {
                        type: "tool_result",
                        content: `<tool_use_error>Error: No such tool available: ${toolUse.name}</tool_use_error>`,
                        is_error: true,
                        tool_use_id: toolUse.id
                    }
                ], { toolUseResult: `Error: No such tool available: ${toolUse.name}` })]
            });
            return;
        }

        const parsedInput = definition.inputSchema?.safeParse(toolUse.input);
        const isConcurrencySafe = parsedInput?.success ? definition.isConcurrencySafe(parsedInput.data) : false;

        this.tools.push({
            id: toolUse.id,
            block: toolUse,
            assistantMessage,
            status: "queued",
            isConcurrencySafe,
            pendingProgress: []
        });

        this.processQueue();
    }

    private canExecuteTool(isConcurrencySafe: boolean): boolean {
        const executing = this.tools.filter(t => t.status === "executing");
        if (executing.length === 0) return true;
        return isConcurrencySafe && executing.every(t => t.isConcurrencySafe);
    }

    private async processQueue() {
        for (const tool of this.tools) {
            if (tool.status !== "queued") continue;
            if (this.canExecuteTool(tool.isConcurrencySafe)) {
                await this.executeTool(tool);
            } else if (!tool.isConcurrencySafe) {
                // Sequential tool must wait for everything before it
                break;
            }
        }
    }

    private createSyntheticErrorMessage(toolUseId: string, reason: string) {
        const content = reason === "user_interrupted"
            ? "Tool use was interrupted by user"
            : "<tool_use_error>Sibling tool call errored</tool_use_error>";

        return createUserMessage([
            {
                type: "tool_result",
                content,
                is_error: true,
                tool_use_id: toolUseId
            }
        ], { toolUseResult: content });
    }

    private getAbortReason(): "sibling_error" | "user_interrupted" | null {
        if (this.hasErrored) return "sibling_error";
        if (this.toolUseContext.abortController?.signal.aborted) return "user_interrupted";
        return null;
    }

    private async executeTool(tool: ToolState) {
        tool.status = "executing";

        // Notify context of in-progress tool
        this.toolUseContext.setInProgressToolUseIDs((ids: Set<string>) => new Set(Array.from(ids).concat([tool.id])));

        const results: any[] = [];
        const modifiers: any[] = [];

        const executionPromise = (async () => {
            const abortReason = this.getAbortReason();
            if (abortReason) {
                tool.results = [this.createSyntheticErrorMessage(tool.id, abortReason)];
                tool.contextModifiers = [];
                tool.status = "completed";
                return;
            }

            const stream = executeToolWithHooks(tool.block, tool.assistantMessage, this.canUseTool, this.toolUseContext);
            let toolErrored = false;

            for await (const chunk of stream as AsyncGenerator<any>) {
                const currentAbortReason = this.getAbortReason();
                if (currentAbortReason && !toolErrored) {
                    results.push(this.createSyntheticErrorMessage(tool.id, currentAbortReason));
                    break;
                }

                if (chunk.message) {
                    if (chunk.message.type === "progress") {
                        tool.pendingProgress.push(chunk.message);
                        if (this.progressAvailableResolve) {
                            this.progressAvailableResolve();
                            this.progressAvailableResolve = undefined;
                        }
                    } else {
                        results.push(chunk.message);
                        // Check if this result is an error (to set hasErrored for siblings)
                        if (chunk.message.type === "user" && Array.isArray(chunk.message.message.content)) {
                            if (chunk.message.message.content.some((c: any) => c.type === "tool_result" && c.is_error)) {
                                this.hasErrored = true;
                                toolErrored = true;
                            }
                        }
                    }
                }

                if (chunk.contextModifier) {
                    modifiers.push(chunk.contextModifier);
                }
            }

            tool.results = results;
            tool.contextModifiers = modifiers;
            tool.status = "completed";

            // If sequential, apply modifiers immediately
            if (!tool.isConcurrencySafe && modifiers.length > 0) {
                for (const mod of modifiers) {
                    this.toolUseContext = mod(this.toolUseContext);
                }
            }
        })();

        tool.promise = executionPromise;
        executionPromise.finally(() => {
            this.processQueue();
        });
    }

    * getCompletedResults(): Generator<{ message?: any }> {
        for (const tool of this.tools) {
            // First yield all pending progress
            while (tool.pendingProgress.length > 0) {
                yield { message: tool.pendingProgress.shift() };
            }

            if (tool.status === "yielded") continue;

            if (tool.status === "completed" && tool.results) {
                tool.status = "yielded";
                for (const result of tool.results) {
                    yield { message: result };
                }
                // Cleanup in-progress ID
                this.toolUseContext.setInProgressToolUseIDs((ids: Set<string>) => {
                    const next = new Set(Array.from(ids));
                    next.delete(tool.id);
                    return next;
                });
            } else if (tool.status === "executing" && !tool.isConcurrencySafe) {
                // Sequential tool block
                break;
            }
        }
    }

    async * getRemainingResults(): AsyncGenerator<{ message?: any }> {
        while (this.hasUnfinishedTools()) {
            await this.processQueue();
            for (const res of Array.from(this.getCompletedResults())) {
                yield res;
            }

            if (this.hasExecutingTools() && !this.hasCompletedResults() && !this.hasPendingProgress()) {
                const executingPromises = this.tools
                    .filter(t => t.status === "executing" && t.promise)
                    .map(t => t.promise!);

                const progressPromise = new Promise<void>(resolve => {
                    this.progressAvailableResolve = resolve;
                });

                if (executingPromises.length > 0) {
                    await Promise.race([...executingPromises, progressPromise]);
                }
            }
        }
        // One last check
        for (const res of Array.from(this.getCompletedResults())) {
            yield res;
        }
    }

    private hasUnfinishedTools(): boolean {
        return this.tools.some(t => t.status !== "yielded");
    }

    private hasExecutingTools(): boolean {
        return this.tools.some(t => t.status === "executing");
    }

    private hasCompletedResults(): boolean {
        return this.tools.some(t => t.status === "completed");
    }

    private hasPendingProgress(): boolean {
        return this.tools.some(t => t.pendingProgress.length > 0);
    }

    getUpdatedContext() {
        return this.toolUseContext;
    }
}
