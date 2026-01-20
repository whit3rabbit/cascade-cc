
import { randomUUID } from "node:crypto";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import {
    createAttachment,
    createMetadataMessage,
    createBannerMessage,
    createAttachmentMessage,
    createToolResultMessage
} from "../terminal/MessageFactory.js";
import { executeStopHooks, getStopHookMessage } from "../terminal/LifecycleService.js";
import {
    preToolExecutionHooks,
    postToolExecutionHooks,
    postToolFailureHooks,
    formatValidationErrors,
    formatHookError
} from "./HookUtils.js";
import { AsyncQueue } from "../../utils/shared/AsyncQueue.js";
import { runParallel } from "../../utils/shared/runParallel.js";
import { mainLoop } from "../loops/MainLoop.js";
import { log } from "../logger/loggerService.js";

const logger = log("hooks");

/**
 * Executes Pre-Stop hooks before the main loop or as part of a tool call. (ag5)
 */
export async function* executePreStopHooks(
    messages: any[],
    systemPrompt: any,
    userContext: any,
    systemContext: any,
    toolUseContext: any,
    querySource: any,
    autoCompactTracking: any,
    fallbackModel: any,
    isSubagent: boolean = false
) {
    const startTime = Date.now();
    const state = {
        messages: [...messages],
        systemPrompt,
        userContext,
        systemContext,
        toolUseContext,
        querySource
    };

    // Prompt suggestion logic (Wv2) would go here

    try {
        const resultMessages: any[] = [];
        const appState = await toolUseContext.getAppState();
        const mode = appState.toolPermissionContext.mode;

        const stopHooks = executeStopHooks(
            mode,
            toolUseContext.abortController.signal,
            undefined,
            isSubagent,
            toolUseContext.agentId,
            toolUseContext,
            messages
        );

        let lastToolUseID = "";
        let toolCallCount = 0;
        let preventContinuation = false;
        let stopReason = "";
        let hasErrors = false;
        const errorMessages: string[] = [];
        const commands: any[] = [];

        for await (const event of stopHooks) {
            if (event.message) {
                yield event.message;
                if (event.message.type === "progress" && event.message.toolUseID) {
                    lastToolUseID = event.message.toolUseID;
                    toolCallCount++;
                    const data = event.message.data;
                    if (data.command) {
                        commands.push({
                            command: data.command,
                            promptText: data.promptText
                        });
                    }
                }
                if (event.message.type === "attachment") {
                    const attachment = event.message.attachment;
                    if (attachment.hookEvent === "Stop" || attachment.hookEvent === "SubagentStop") {
                        if (attachment.type === "hook_non_blocking_error") {
                            errorMessages.push(attachment.stderr || `Exit code ${attachment.exitCode}`);
                            hasErrors = true;
                        } else if (attachment.type === "hook_error_during_execution") {
                            errorMessages.push(attachment.content);
                            hasErrors = true;
                        } else if (attachment.type === "hook_success") {
                            if (attachment.stdout?.trim() || attachment.stderr?.trim()) {
                                hasErrors = true;
                            }
                        }
                    }
                }
            }

            if (event.blockingError) {
                const errorMsg = createMetadataMessage({
                    content: getStopHookMessage(event.blockingError),
                    isMeta: true
                });
                resultMessages.push(errorMsg);
                yield errorMsg;
                hasErrors = true;
                errorMessages.push(event.blockingError.blockingError);
            }

            if (event.preventContinuation) {
                preventContinuation = true;
                stopReason = event.stopReason || "Stop hook prevented continuation";
                yield createAttachmentMessage({
                    type: "hook_stopped_continuation",
                    message: stopReason,
                    hookName: "Stop",
                    toolUseID: lastToolUseID,
                    hookEvent: "Stop"
                });
            }

            if (toolUseContext.abortController.signal.aborted) {
                logTelemetryEvent("tengu_pre_stop_hooks_cancelled", {
                    queryChainId: toolUseContext.queryTracking?.chainId,
                    queryDepth: toolUseContext.queryTracking?.depth
                });
                yield { type: "done", toolUse: false };
                return;
            }
        }

        if (toolCallCount > 0 && errorMessages.length > 0) {
            toolUseContext.addNotification?.({
                key: "stop-hook-error",
                text: "Stop hook error occurred · ctrl+o to see",
                priority: "immediate"
            });
        }

        if (preventContinuation) return;

        if (resultMessages.length > 0) {
            yield* mainLoop({
                messages: [...messages, ...resultMessages],
                systemPrompt,
                userContext,
                systemContext,
                canUseTool: () => Promise.resolve({ behavior: "allow" }), // placeholder
                toolUseContext,
                autoCompactTracking,
                fallbackModel,
                stopHookActive: true,
                querySource
            });
        }
    } catch (err) {
        const duration = Date.now() - startTime;
        logTelemetryEvent("tengu_stop_hook_error", {
            duration,
            queryChainId: toolUseContext.queryTracking?.chainId,
            queryDepth: toolUseContext.queryTracking?.depth
        });
        yield createBannerMessage(`Stop hook failed: ${err instanceof Error ? err.message : String(err)}`, "warning");
    }
}

/**
 * Groups tools by concurrency safety for mixed sequential/parallel execution. (og5)
 */
function groupToolsByConcurrency(toolUses: any[], context: any) {
    return toolUses.reduce((blocks: any[], toolUse: any) => {
        const tool = context.options.tools.find((t: any) => t.name === toolUse.name);
        // Only validate if input exists
        const isSafe = tool?.isConcurrencySafe ? Boolean(tool.isConcurrencySafe(toolUse.input)) : false;

        if (isSafe && blocks[blocks.length - 1]?.isConcurrencySafe) {
            blocks[blocks.length - 1].tools.push(toolUse);
        } else {
            blocks.push({
                isConcurrencySafe: isSafe,
                tools: [toolUse]
            });
        }
        return blocks;
    }, []);
}

/**
 * Orchestrates concurrent or sequential tool execution. (fF0)
 */
export async function* processToolsConcurrently(
    toolUses: any[],
    assistantMessages: any[],
    canUseTool: any,
    toolUseContext: any
) {
    let currentContext = toolUseContext;
    const executionBlocks = groupToolsByConcurrency(toolUses, currentContext);

    for (const block of executionBlocks) {
        if (block.isConcurrencySafe) {
            const contextModifiers: Record<string, any[]> = {};
            const parallelResults = runParallel(block.tools.map((toolUse: any) => {
                return async function* () {
                    toolUseContext.setInProgressToolUseIDs((ids: Set<string>) => new Set(Array.from(ids).concat([toolUse.id])));
                    const findAssistantMessage = assistantMessages.find(m =>
                        m.message.content.some((c: any) => c.type === "tool_use" && c.id === toolUse.id)
                    );
                    yield* executeToolWithHooks(toolUse, findAssistantMessage, canUseTool, toolUseContext);
                    toolUseContext.setInProgressToolUseIDs((ids: Set<string>) => {
                        const next = new Set(Array.from(ids));
                        next.delete(toolUse.id);
                        return next;
                    });
                };
            }), 10);

            for await (const result of (parallelResults as any)) {
                if ((result as any).contextModifier) {
                    const { toolUseID, modifyContext } = (result as any).contextModifier;
                    if (!contextModifiers[toolUseID]) contextModifiers[toolUseID] = [];
                    contextModifiers[toolUseID].push(modifyContext);
                }
                yield {
                    message: (result as any).message,
                    newContext: currentContext
                };
            }

            for (const toolUse of block.tools) {
                const modifiers = contextModifiers[toolUse.id];
                if (modifiers) {
                    for (const modify of modifiers) {
                        currentContext = modify(currentContext);
                    }
                }
            }
            yield { newContext: currentContext };
        } else {
            for (const toolUse of block.tools) {
                currentContext.setInProgressToolUseIDs((ids: Set<string>) => new Set(Array.from(ids).concat([toolUse.id])));
                const findAssistantMessage = assistantMessages.find(m =>
                    m.message.content.some((c: any) => c.type === "tool_use" && c.id === toolUse.id)
                );
                for await (const result of executeToolWithHooks(toolUse, findAssistantMessage, canUseTool, currentContext)) {
                    if ((result as any).contextModifier) {
                        currentContext = (result as any).contextModifier.modifyContext(currentContext);
                    }
                    yield {
                        message: (result as any).message,
                        newContext: currentContext
                    };
                }
                currentContext.setInProgressToolUseIDs((ids: Set<string>) => {
                    const next = new Set(Array.from(ids));
                    next.delete(toolUse.id);
                    return next;
                });
            }
        }
    }
}

/**
 * Handles tool execution with pre/post hooks. (CX1)
 */
export async function* executeToolWithHooks(
    toolUse: any,
    assistantMessage: any,
    canUseTool: any,
    context: any
) {
    const toolName = toolUse.name;
    const tool = context.options.tools.find((t: any) => t.name === toolName);
    const messageID = assistantMessage.message.id;
    const requestId = assistantMessage.requestId;
    const isMcp = toolName.startsWith("mcp__");

    if (!tool) {
        logTelemetryEvent("tengu_tool_use_error", {
            error: `No such tool available: ${toolName}`,
            toolName,
            toolUseID: toolUse.id,
            isMcp,
            queryChainId: context.queryTracking?.chainId,
            queryDepth: context.queryTracking?.depth,
            requestId
        });
        yield {
            message: createMetadataMessage({
                content: [{
                    type: "tool_result",
                    content: `<tool_use_error>Error: No such tool available: ${toolName}</tool_use_error>`,
                    is_error: true,
                    tool_use_id: toolUse.id
                }],
                toolUseResult: `Error: No such tool available: ${toolName}`
            })
        };
        return;
    }

    try {
        if (context.abortController.signal.aborted) {
            yield {
                message: createMetadataMessage({
                    content: [{
                        type: "tool_result",
                        content: `<tool_use_error>Interrupted by user</tool_use_error>`,
                        is_error: true,
                        tool_use_id: toolUse.id
                    }],
                    toolUseResult: "Interrupted"
                })
            };
            return;
        }

        const queue = new AsyncQueue();
        performToolCall(tool, toolUse.id, toolUse.input, context, canUseTool, assistantMessage, messageID, requestId, isMcp ? "stdio" : undefined, (progress: any) => {
            queue.enqueue({
                message: {
                    type: "progress",
                    toolUseID: progress.toolUseID,
                    parentToolUseID: toolUse.id,
                    data: progress.data
                }
            });
        }).then((results: any[]) => {
            for (const res of results) queue.enqueue(res);
        }).catch((err: any) => {
            queue.error(err);
        }).finally(() => {
            queue.done();
        });

        for await (const event of queue) {
            yield event;
        }
    } catch (err) {
        const errorText = formatHookError(err);
        yield {
            message: createMetadataMessage({
                content: [{
                    type: "tool_result",
                    content: `<tool_use_error>${errorText}</tool_use_error>`,
                    is_error: true,
                    tool_use_id: toolUse.id
                }],
                toolUseResult: errorText
            })
        };
    }
}

/**
 * Core tool call logic, including Zod validation and permission checks. (Au5)
 */
async function performToolCall(
    tool: any,
    toolUseID: string,
    input: any,
    context: any,
    canUseTool: any,
    assistantMessage: any,
    messageID: string,
    requestId: string,
    mcpServerType: any,
    onProgress: (p: any) => void
): Promise<any[]> {
    const validation = tool.inputSchema.safeParse(input);
    if (!validation.success) {
        return [{
            message: createMetadataMessage({
                content: [{
                    type: "tool_result",
                    content: `<tool_use_error>InputValidationError: ${formatValidationErrors(tool.name, validation.error)}</tool_use_error>`,
                    is_error: true,
                    tool_use_id: toolUseID
                }],
                toolUseResult: `InputValidationError: ${validation.error.message}`
            })
        }];
    }

    let finalInput = validation.data;
    let stopContinuation = false;
    let stopReason = "";
    let hookPermissionResult: any = undefined;
    const messages: any[] = [];

    // Pre-execution hooks
    for await (const hookResult of preToolExecutionHooks(context, tool, finalInput, toolUseID, messageID, requestId, mcpServerType)) {
        if (hookResult.type === "message" && hookResult.message) {
            if (hookResult.message.message?.type === "progress") {
                onProgress(hookResult.message.message);
            } else {
                messages.push(hookResult.message);
            }
        } else if (hookResult.type === "hookPermissionResult") {
            hookPermissionResult = hookResult.hookPermissionResult;
        } else if (hookResult.type === "preventContinuation") {
            stopContinuation = Boolean(hookResult.shouldPreventContinuation);
        } else if (hookResult.type === "stopReason") {
            stopReason = hookResult.stopReason || "";
        } else if (hookResult.type === "stop") {
            messages.push({
                message: createMetadataMessage({
                    content: [{
                        type: "tool_result",
                        content: `<tool_use_error>Interrupted</tool_use_error>`,
                        is_error: true,
                        tool_use_id: toolUseID
                    }],
                    toolUseResult: `Interrupted`
                })
            });
            return messages;
        }
    }

    // Permission check
    let decision: any;
    if (hookPermissionResult?.behavior === "allow" && !tool.requiresUserInteraction?.()) {
        decision = hookPermissionResult;
    } else {
        decision = await canUseTool(tool, finalInput, context, assistantMessage, toolUseID, hookPermissionResult?.behavior === "ask" ? hookPermissionResult : undefined);
    }

    if (decision.behavior !== "allow") {
        let message = decision.message;
        if (stopContinuation && !message) message = `Execution stopped by PreToolUse hook${stopReason ? `: ${stopReason}` : ""}`;
        messages.push({
            message: createMetadataMessage({
                content: [{
                    type: "tool_result",
                    content: message,
                    is_error: true,
                    tool_use_id: toolUseID
                }],
                toolUseResult: `Error: ${message}`
            })
        });
        return messages;
    }

    finalInput = decision.updatedInput || finalInput;
    const callStartTime = Date.now();
    try {
        const result = await tool.call(finalInput, {
            ...context,
            userModified: decision.userModified ?? false
        }, canUseTool, assistantMessage, (p: any) => {
            onProgress({ toolUseID: p.toolUseID, data: p.data });
        });

        const duration = Date.now() - callStartTime;
        const toolOutput = result.data;
        const resultMessageContent: any[] = [await createToolResultMessage(tool, toolOutput, toolUseID)];

        if (decision.acceptFeedback) {
            resultMessageContent.push({ type: "text", text: decision.acceptFeedback });
        }

        messages.push({
            message: createMetadataMessage({
                content: resultMessageContent,
                toolUseResult: toolOutput
            }),
            contextModifier: result.contextModifier ? {
                toolUseID,
                modifyContext: result.contextModifier
            } : undefined
        });

        // Post-execution hooks
        for await (const postHookResult of postToolExecutionHooks(context, tool, toolUseID, messageID, toolOutput, requestId, mcpServerType)) {
            if ("updatedMCPToolOutput" in postHookResult) {
                // MCP output update logic if needed
            } else if (postHookResult.message) {
                messages.push(postHookResult);
            }
        }

        if (result.newMessages) {
            for (const msg of result.newMessages) messages.push({ message: msg });
        }

        return messages;
    } catch (err) {
        const errorText = formatHookError(err);
        const failureMessages = [];
        for await (const failHookResult of postToolFailureHooks(context, tool, toolUseID, messageID, finalInput, errorText, false, requestId, mcpServerType)) {
            failureMessages.push(failHookResult);
        }

        return [{
            message: createMetadataMessage({
                content: [{
                    type: "tool_result",
                    content: `Error: ${errorText}`,
                    is_error: true,
                    tool_use_id: toolUseID
                }],
                toolUseResult: `Error: ${errorText}`
            })
        }, ...failureMessages];
    }
}

/**
 * Executes Pre-Compact hooks. (_H0)
 */
export async function executePreCompactHooks(
    trigger: string,
    customInstructions?: string,
    signal?: AbortSignal
) {
    // Logic for running PreCompact hooks
    return {
        newCustomInstructions: customInstructions,
        userDisplayMessage: undefined as string | undefined
    };
}
