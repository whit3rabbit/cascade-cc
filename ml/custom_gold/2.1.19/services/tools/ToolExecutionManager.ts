/**
 * File: src/services/tools/ToolExecutionManager.ts
 * Role: Manages parallel and serial execution of tools, including queuing and progress tracking.
 */

import { randomUUID } from "node:crypto";

/**
 * Constant for error content when the user interrupts tool execution.
 */
export const USER_INTERRUPTED_ERROR_CONTENT = "<tool_use_error>User rejected tool use</tool_use_error>";

export interface ToolBlock {
    id: string;
    name: string;
    input: any;
}

export interface AssistantMessage {
    uuid: string;
    [key: string]: any;
}

export interface ToolResult {
    content: any[];
    toolUseResult: any;
    sourceToolAssistantUUID: string;
    isMeta: boolean;
    is_error?: boolean;
}

export interface ToolUseContext {
    abortController?: AbortController;
    setInProgressToolUseIDs?: (fn: (prev: Set<string>) => Set<string>) => void;
    removeInProgressToolUseID?: (id: string) => void;
    queryTracking?: {
        chainId: string;
        depth: number;
    };
    [key: string]: any;
}

export interface ToolDefinition {
    name: string;
    isConcurrencySafe?: (input: any) => boolean;
    [key: string]: any;
}

export interface ManagedTool {
    id: string;
    block: ToolBlock;
    assistantMessage: AssistantMessage;
    status: "queued" | "executing" | "completed" | "yielded";
    isConcurrencySafe: boolean;
    pendingProgress: any[];
    results?: any[];
    contextModifiers?: any[];
    promise?: Promise<void>;
}

/**
 * ToolExecutionManager handles the lifecycle of tool calls within a single assistant response.
 */
export class ToolExecutionManager {
    public toolDefinitions: ToolDefinition[];
    public canUseTool: (name: string, input: any, context: any) => Promise<any>;
    public toolUseContext: ToolUseContext;
    public executeToolFn: (block: ToolBlock, message: AssistantMessage, canUse: any, context: any) => AsyncGenerator<any>;
    public tools: ManagedTool[];
    public hasErrored: boolean;
    public discarded: boolean;
    private progressAvailableResolve?: () => void;

    /**
     * @param toolDefinitions - Available tool definitions.
     * @param canUseTool - Permission check function.
     * @param toolUseContext - Shared context for tool execution.
     * @param executeToolFn - The function that actually invokes the tool (generator).
     */
    constructor(
        toolDefinitions: ToolDefinition[],
        canUseTool: (name: string, input: any, context: any) => Promise<any>,
        toolUseContext: ToolUseContext,
        executeToolFn: (block: ToolBlock, message: AssistantMessage, canUse: any, context: any) => AsyncGenerator<any>
    ) {
        this.toolDefinitions = toolDefinitions;
        this.canUseTool = canUseTool;
        this.toolUseContext = toolUseContext;
        this.executeToolFn = executeToolFn;
        this.tools = [];
        this.hasErrored = false;
        this.discarded = false;
    }

    /**
     * Discards the current execution (e.g., due to a streaming fallback).
     */
    discard(): void {
        this.discarded = true;
    }

    /**
     * Adds a tool call to the execution queue.
     */
    addTool(toolBlock: ToolBlock, assistantMessage: AssistantMessage): void {
        const toolDefinition = this.toolDefinitions.find(tool => tool.name === toolBlock.name);

        if (!toolDefinition) {
            const errorResult = this.createToolUseError(toolBlock.id, `Error: No such tool available: ${toolBlock.name}`, assistantMessage);
            this.tools.push({
                id: toolBlock.id,
                block: toolBlock,
                assistantMessage,
                status: "completed",
                isConcurrencySafe: true,
                pendingProgress: [],
                results: [errorResult]
            });
            return;
        }

        // Validate plan/concurrency safety
        const isConcurrencySafe = typeof toolDefinition.isConcurrencySafe === 'function'
            ? toolDefinition.isConcurrencySafe(toolBlock.input)
            : false;

        this.tools.push({
            id: toolBlock.id,
            block: toolBlock,
            assistantMessage,
            status: "queued",
            isConcurrencySafe: isConcurrencySafe,
            pendingProgress: []
        });

        this.processQueue();
    }

    /**
     * Checks if a new tool can start executing.
     */
    canExecuteTool(isConcurrencySafe: boolean): boolean {
        const executingTools = this.tools.filter(t => t.status === "executing");
        if (executingTools.length === 0) return true;

        // Parallel execution allowed only if all participating tools are concurrency-safe.
        return isConcurrencySafe && executingTools.every(t => t.isConcurrencySafe);
    }

    /**
     * Processes the queue, starting tools as concurrency rules allow.
     */
    async processQueue(): Promise<void> {
        for (const tool of this.tools) {
            if (tool.status !== "queued") continue;

            if (this.canExecuteTool(tool.isConcurrencySafe)) {
                await this.executeTool(tool);
            } else if (!tool.isConcurrencySafe) {
                // If we hit a synchronous tool, we must wait for current tools to finish.
                break;
            }
        }
    }

    /**
     * Creates a standardized tool error result.
     */
    createToolUseError(toolUseId: string, toolUseResult: any, assistantMessage: AssistantMessage, customContent: string | null = null): ToolResult {
        return {
            content: [{
                type: "tool_result",
                content: customContent || `<tool_use_error>${toolUseResult}</tool_use_error>`,
                is_error: true,
                toolUseId: toolUseId
            }],
            toolUseResult: toolUseResult,
            sourceToolAssistantUUID: assistantMessage.uuid,
            isMeta: true
        };
    }

    /**
     * Determines if and why the execution should be aborted.
     */
    getAbortReason(): "streaming_fallback" | "sibling_error" | "user_interrupted" | null {
        if (this.discarded) return "streaming_fallback";
        if (this.hasErrored) return "sibling_error";
        if (this.toolUseContext.abortController?.signal.aborted) return "user_interrupted";
        return null;
    }

    /**
     * Executes a tool via the provided execution function.
     */
    async executeTool(tool: ManagedTool): Promise<void> {
        tool.status = "executing";

        // Track in-progress ID
        if (this.toolUseContext.setInProgressToolUseIDs) {
            this.toolUseContext.setInProgressToolUseIDs(prev => new Set([...prev, tool.id]));
        }

        const results: any[] = [];
        const contextModifiers: any[] = [];

        const executionPromise = (async () => {
            const earlyAbort = this.getAbortReason();
            if (earlyAbort) {
                results.push(this.createSyntheticErrorMessage(tool.id, earlyAbort, tool.assistantMessage));
                tool.results = results;
                tool.status = "completed";
                return;
            }

            try {
                const generator = this.executeToolFn(tool.block, tool.assistantMessage, this.canUseTool, this.toolUseContext);

                for await (const result of generator) {
                    const lateAbort = this.getAbortReason();
                    if (lateAbort) {
                        results.push(this.createSyntheticErrorMessage(tool.id, lateAbort, tool.assistantMessage));
                        break;
                    }

                    if (result.message) {
                        if (result.message.type === "progress") {
                            tool.pendingProgress.push(result.message);
                            if (this.progressAvailableResolve) {
                                this.progressAvailableResolve();
                                this.progressAvailableResolve = undefined;
                            }
                        } else {
                            results.push(result.message);
                            // Check if result contains an error to stop parallel siblings if needed
                            if (result.message.is_error) this.hasErrored = true;
                        }
                    }

                    if (result.contextModifier) {
                        contextModifiers.push(result.contextModifier.modifyContext);
                    }
                }
            } catch (error: any) {
                console.error(`Tool execution failed for ${tool.id}:`, error);
                results.push(this.createToolUseError(tool.id, `Exec error: ${error.message}`, tool.assistantMessage));
            }

            tool.results = results;
            tool.contextModifiers = contextModifiers;
            tool.status = "completed";

            // Apply context modifiers if single-threaded
            if (!tool.isConcurrencySafe && contextModifiers.length > 0) {
                for (const mod of contextModifiers) {
                    this.toolUseContext = mod(this.toolUseContext);
                }
            }
        })();

        tool.promise = executionPromise;
        executionPromise.finally(() => this.processQueue());
    }

    createSyntheticErrorMessage(id: string, reason: string, assistantMessage: AssistantMessage): ToolResult {
        const reasons: Record<string, { content: string, msg: string }> = {
            "user_interrupted": { content: USER_INTERRUPTED_ERROR_CONTENT, msg: "User rejected tool use" },
            "streaming_fallback": { content: "<tool_use_error>Streaming fallback</tool_use_error>", msg: "Discarded" },
            "sibling_error": { content: "<tool_use_error>Sibling tool failed</tool_use_error>", msg: "Sibling failed" }
        };
        const r = reasons[reason] || { content: "<tool_use_error>Unknown</tool_use_error>", msg: "Unknown" };
        return this.createToolUseError(id, r.msg, assistantMessage, r.content);
    }

    /**
     * Yields results as they become available.
     */
    *getCompletedResults(): Generator<{ message: any }> {
        if (this.discarded) return;

        for (const tool of this.tools) {
            while (tool.pendingProgress.length > 0) {
                yield { message: tool.pendingProgress.shift() };
            }

            if (tool.status === "yielded") continue;

            if (tool.status === "completed" && tool.results) {
                tool.status = "yielded";
                for (const res of tool.results) {
                    yield { message: res };
                }
                // Cleanup in-progress ID
                if (this.toolUseContext.removeInProgressToolUseID) {
                    this.toolUseContext.removeInProgressToolUseID(tool.id);
                }
            } else if (tool.status === "executing" && !tool.isConcurrencySafe) {
                break;
            }
        }
    }

    /**
     * Asynchronously waits for all tools to finish, yielding results.
     */
    async *getRemainingResults(): AsyncGenerator<{ message: any }> {
        while (this.tools.some(t => t.status !== "yielded")) {
            await this.processQueue();
            for (const res of this.getCompletedResults()) yield res;

            const executing = this.tools.filter(t => t.status === "executing" && t.promise) as (ManagedTool & { promise: Promise<void> })[];
            const completed = this.tools.some(t => t.status === "completed");
            const hasProgress = this.tools.some(t => t.pendingProgress.length > 0);

            if (executing.length > 0 && !completed && !hasProgress) {
                const progressPromise = new Promise<void>(r => { this.progressAvailableResolve = r; });
                await Promise.race([...executing.map(t => t.promise), progressPromise]);
            } else if (executing.length === 0 && this.tools.some(t => t.status === "queued")) {
                // Just in case nothing is executing but something is queued
                await this.processQueue();
            } else if (executing.length === 0 && !completed && !hasProgress) {
                // Nothing left to do
                break;
            }

            // Artificial delay to prevent infinite loop if status doesn't change immediately
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

/**
 * Utility for creating a sub-context for tool execution.
 */
export function forkToolUseContext(originalContext: any, overrides: any = {}): ToolUseContext {
    return {
        ...originalContext,
        ...overrides,
        abortController: new AbortController(),
        queryTracking: {
            chainId: randomUUID(),
            depth: (originalContext.queryTracking?.depth ?? -1) + 1
        }
    };
}
