import { useState, useCallback, useRef } from "react";
import { randomUUID } from "node:crypto";
import { useAppState, getAppState } from "../../contexts/AppStateContext.js";
import {
    createUserMessage,
    createAssistantMessage,
    createMetadataMessage,
    createBannerMessage,
    createAttachmentMessage,
    normalizeMessages,
    createToolResultMessage
} from "./MessageFactory.js";
import { log, logError, logException } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { streamAnthropic } from "./anthropicStream.js";
import {
    compactConversation,
    microCompactMessages
} from "../history/ConversationHistoryManager.js";
import { executePreStopHooks } from "../hooks/HookExecutor.js";
import { ToolExecutionQueue } from "../loops/ToolExecutionQueue.js";

const logger = log("terminal");

// Constants
const MAX_CONCURRENCY = parseInt(process.env.CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY || "", 10) || 10;

export interface MainLoopOptions {
    messages: any[];
    systemPrompt: string;
    userContext?: string;
    systemContext?: string;
    canUseTool: (tool: any, input: any, context: any) => Promise<any>;
    toolUseContext: any;
    autoCompactTracking?: any;
    fallbackModel?: string;
    stopHookActive?: boolean;
    querySource?: string;
    maxOutputTokensOverride?: number;
}

/**
 * The core agent execution loop. This is an async generator that yields
 * events as Claude responds, uses tools, and transitions state.
 * Deobfuscated from ew in chunk_457.ts.
 */
export async function* mainLoop(options: MainLoopOptions): AsyncGenerator<any> {
    const {
        messages,
        systemPrompt,
        userContext,
        systemContext,
        canUseTool,
        fallbackModel,
        stopHookActive = false,
        querySource = "main",
        maxOutputTokensOverride
    } = options;
    let { toolUseContext, autoCompactTracking } = options;

    yield { type: "stream_request_start" };

    if (!toolUseContext.agentId) {
        logTelemetryEvent("query_started");
    }

    const queryTracking = toolUseContext.queryTracking ? {
        chainId: toolUseContext.queryTracking.chainId,
        depth: toolUseContext.queryTracking.depth + 1
    } : {
        chainId: randomUUID(),
        depth: 0
    };

    const chainId = queryTracking.chainId;
    toolUseContext = { ...toolUseContext, queryTracking };

    // 1. Prepare messages and handle compaction (CT2/Vd in chunk_457)
    const { messages: normalizedMessages } = await microCompactMessages(messages, undefined, toolUseContext);
    let currentMessages = normalizedMessages;

    const { compactionResult } = await compactConversation(currentMessages, toolUseContext, querySource);
    let tracking = autoCompactTracking;

    if (compactionResult) {
        logTelemetryEvent("tengu_auto_compact_succeeded", {
            originalMessageCount: messages.length,
            compactedMessageCount: compactionResult.summaryMessages.length + (compactionResult.attachments?.length ?? 0) + (compactionResult.hookResults?.length ?? 0),
            preCompactTokenCount: compactionResult.preCompactTokenCount,
            postCompactTokenCount: compactionResult.postCompactTokenCount,
            queryChainId: chainId,
            queryDepth: queryTracking.depth
        });

        if (!tracking?.compacted) {
            tracking = {
                compacted: true,
                turnId: randomUUID(),
                turnCounter: 0
            };
        }

        const newMessages = [
            compactionResult.boundaryMarker,
            ...compactionResult.summaryMessages,
            ...(compactionResult.attachments ?? []),
            ...(compactionResult.hookResults ?? []),
            ...(compactionResult.messagesToKeep ?? [])
        ];

        for (const msg of newMessages) {
            yield msg;
        }
        currentMessages = newMessages;
    }

    toolUseContext = { ...toolUseContext, messages: currentMessages };

    // 2. Prepare for execution
    const assistantMessages: any[] = [];
    const toolResults: any[] = [];
    const executionQueue = new ToolExecutionQueue(toolUseContext.options.tools, canUseTool, toolUseContext);

    const appState = await toolUseContext.getAppState();
    const permissionMode = appState.toolPermissionContext.mode;

    // Determine model (D1A)
    let activeModel = getModel({
        permissionMode,
        mainLoopModel: toolUseContext.options.mainLoopModel,
        messages: currentMessages,
        fallbackModel
    });

    const fullSystemPrompt = formatSystemPrompt(systemPrompt, systemContext);

    // 3. Execution Loop
    let shouldContinue = true;
    while (shouldContinue) {
        shouldContinue = false;
        try {
            const stream = streamAnthropic({
                messages: prepareMessagesForApi(currentMessages, userContext),
                systemPrompt: fullSystemPrompt,
                maxThinkingTokens: toolUseContext.options.maxThinkingTokens,
                tools: toolUseContext.options.tools,
                signal: toolUseContext.abortController.signal,
                options: {
                    model: activeModel,
                    querySource,
                    agentId: toolUseContext.agentId,
                    queryTracking,
                    maxOutputTokensOverride,
                    mcpTools: appState.mcp?.tools ?? []
                }
            });

            for await (const event of stream) {
                yield event;
                if (event.type === "assistant") {
                    assistantMessages.push(event);
                    const toolUses = event.message.content.filter((c: any) => c.type === "tool_use");
                    for (const toolUse of toolUses) {
                        executionQueue.addTool(toolUse, event);
                    }
                }

                // Yield completed tool results during streaming (tengu_streaming_tool_execution2)
                for (const res of Array.from(executionQueue.getCompletedResults())) {
                    if (res.message) {
                        yield res.message;
                        if (res.message.type === "user") {
                            toolResults.push(res.message);
                        }
                    }
                }
            }
        } catch (error: any) {
            if (error.type === "overloaded" && fallbackModel && activeModel !== fallbackModel) {
                activeModel = fallbackModel;
                shouldContinue = true;
                logTelemetryEvent("tengu_model_fallback_triggered", {
                    original_model: activeModel,
                    fallback_model: fallbackModel,
                    queryChainId: chainId,
                    queryDepth: queryTracking.depth
                });
                // Send failure markers for orphan messages (kF0)
                for (const msg of Array.from(createToolResultErrorMessages(assistantMessages, "Model fallback triggered"))) {
                    yield msg;
                }
                assistantMessages.length = 0;
                continue;
            }

            logError("MainLoop", error);
            logTelemetryEvent("tengu_query_error", {
                assistantMessages: assistantMessages.length,
                queryChainId: chainId,
                queryDepth: queryTracking.depth
            });
            for (const msg of Array.from(createToolResultErrorMessages(assistantMessages, error.message || String(error)))) {
                yield msg;
            }
            throw error;
        }
    }

    // 4. Handle interruption or remaining tools
    if (toolUseContext.abortController.signal.aborted) {
        for await (const res of executionQueue.getRemainingResults()) {
            if (res.message) yield res.message;
        }
        for (const msg of Array.from(createToolResultErrorMessages(assistantMessages, "Interrupted by user"))) {
            yield msg;
        }
        return;
    }

    const allToolUses = assistantMessages.flatMap(m => m.message.content.filter((c: any) => c.type === "tool_use"));

    if (assistantMessages.length === 0 || allToolUses.length === 0) {
        // executePostStopHooks (ag5)
        if (!stopHookActive) {
            yield* executePreStopHooks(
                [...currentMessages, ...assistantMessages, ...toolResults],
                systemPrompt,
                userContext,
                systemContext,
                toolUseContext,
                querySource,
                tracking,
                fallbackModel
            );

            // Generate suggestions (Wv2)
            generatePromptSuggestion({
                messages: [...currentMessages, ...assistantMessages, ...toolResults],
                toolUseContext,
                querySource
            }).catch(err => logError("MainLoopSuggestion", err));
        }
        return;
    }

    // Process remaining tools
    let processStopped = false;
    for await (const res of executionQueue.getRemainingResults()) {
        if (res.message) {
            yield res.message;
            if (res.message.type === "user") {
                toolResults.push(res.message);
            }
            if (res.message.type === "attachment" && res.message.attachment?.type === "hook_stopped_continuation") {
                processStopped = true;
            }
        }
    }

    if (processStopped || toolUseContext.abortController.signal.aborted) return;

    if (tracking?.compacted) {
        tracking.turnCounter++;
    }

    // 5. Handle Steering / Queued Commands (FHA / eY2)
    const nextAppState = await toolUseContext.getAppState();
    const steeringAttachments: any[] = [];
    if (nextAppState.queuedCommands?.length > 0) {
        for (const cmd of nextAppState.queuedCommands) {
            steeringAttachments.push(createUserMessage(cmd));
        }
        await toolUseContext.setAppState((s: any) => ({ ...s, queuedCommands: [] }));
    }

    if (toolUseContext.pendingSteeringAttachments) {
        steeringAttachments.push(...toolUseContext.pendingSteeringAttachments);
    }

    // 6. Recurse
    yield* mainLoop({
        ...options,
        messages: [...currentMessages, ...assistantMessages, ...toolResults, ...steeringAttachments],
        toolUseContext: {
            ...executionQueue.getUpdatedContext(),
            queryTracking,
            pendingSteeringAttachments: steeringAttachments.length > 0 ? steeringAttachments : undefined
        },
        autoCompactTracking: tracking
    });
}

/**
 * Logic to decide which model to use. (D1A)
 */
function getModel(options: {
    permissionMode: string,
    mainLoopModel?: string,
    messages: any[],
    fallbackModel?: string
}): string {
    return options.mainLoopModel || options.fallbackModel || "claude-3-5-sonnet-20241022";
}

/**
 * Combines system prompt with context. (dy2)
 */
function formatSystemPrompt(basePrompt: string, systemContext?: string): string {
    if (!systemContext) return basePrompt;
    return `${basePrompt}\n\n<system_context>\n${systemContext}\n</system_context>`;
}

/**
 * Prepares message array for API call by injecting user context. (v9A)
 */
function prepareMessagesForApi(messages: any[], userContext?: string): any[] {
    if (!userContext) return messages;

    // Deobfuscated logic from v9A: find the first user message and wrap its content
    return messages.map((msg, index) => {
        if (msg.type === "user" && index === messages.findIndex(m => m.type === "user")) {
            const originalContent = typeof msg.message.content === 'string'
                ? msg.message.content
                : (msg.message.content?.[0]?.text || "");

            return {
                ...msg,
                message: {
                    ...msg.message,
                    content: [
                        { type: "text", text: `<user_context>\n${userContext}\n</user_context>\n\n${originalContent}` }
                    ]
                }
            };
        }
        return msg;
    });
}

/**
 * Creates error results for tools that were called but never finished. (kF0)
 */
function* createToolResultErrorMessages(assistantMessages: any[], error: string) {
    for (const msg of assistantMessages) {
        const toolUses = msg.message.content.filter((c: any) => c.type === "tool_use");
        for (const toolUse of toolUses) {
            yield createMetadataMessage({
                content: [
                    {
                        type: "tool_result",
                        content: `<tool_use_error>${error}</tool_use_error>`,
                        is_error: true,
                        tool_use_id: toolUse.id
                    }
                ],
                toolUseResult: error,
                requestId: msg.requestId
            });
        }
    }
}

let suggestionAbortController: AbortController | null = null;

/**
 * Generates prompt suggestions in the background. (Wv2)
 */
export async function generatePromptSuggestion(options: {
    messages: any[],
    toolUseContext: any,
    querySource?: string
}) {
    if (options.querySource !== "repl_main_thread") return;

    const state = await options.toolUseContext.getAppState();
    if (!state.promptSuggestionEnabled) return;

    if (suggestionAbortController) suggestionAbortController.abort();

    if (state.pendingWorkerRequest || state.pendingSandboxRequest) return;
    if (state.elicitation?.queue?.length > 0) return;
    if (state.toolPermissionContext.mode === "plan") return;

    if (options.messages.filter(m => m.type === "assistant").length < 2) return;

    suggestionAbortController = new AbortController();
    const signal = suggestionAbortController.signal;

    try {
        // Implementation would involve a call to haiku (ug5 in chunk_457)
        // For now, we keep the trigger logic accurate.
    } catch (err) {
        // ignore
    } finally {
        if (suggestionAbortController?.signal === signal) suggestionAbortController = null;
    }
}

/**
 * Orchestrator hook for the terminal.
 * Deobfuscated from mDA in chunk_838.ts.
 */
export function useAgentMainLoop(initialMessages: any[] = []) {
    const [messages, setMessages] = useState<any[]>(initialMessages);
    const [isResponding, setIsResponding] = useState(false);
    const [responseLength, setResponseLength] = useState(0);
    const [streamMode, setStreamMode] = useState<"responding" | "thinking" | "tool-input" | "requesting">("responding");
    const abortControllerRef = useRef<AbortController | null>(null);

    const onQuery = useCallback(async (input: string) => {
        if (isResponding) return;

        setIsResponding(true);
        setResponseLength(0);
        setStreamMode("requesting");

        const userMsg = createUserMessage(input);
        const newMessages = [...messages, userMsg];
        setMessages(newMessages);

        abortControllerRef.current = new AbortController();

        try {
            const state = getAppState();
            const options: MainLoopOptions = {
                messages: newMessages,
                systemPrompt: state.systemPrompt || "You are Claude Code, a helpful assistant.",
                canUseTool: async (tool, input, context) => ({ behavior: "allow" }), // Permission logic
                toolUseContext: {
                    options: {
                        tools: state.availableTools || [],
                        mainLoopModel: state.mainLoopModel
                    },
                    abortController: abortControllerRef.current,
                    getAppState: () => Promise.resolve(getAppState()),
                    setAppState: (updater: any) => { /* logic to update app state */ },
                    setInProgressToolUseIDs: (updater: any) => { /* logic */ }
                }
            };

            for await (const event of mainLoop(options)) {
                processMainLoopEvent(event, setMessages, setResponseLength, setStreamMode);
            }

        } catch (error: any) {
            logError("AgentMainLoop", error);
        } finally {
            setIsResponding(false);
            abortControllerRef.current = null;
        }
    }, [messages, isResponding]);

    const onInterrupt = useCallback(() => {
        abortControllerRef.current?.abort();
        setIsResponding(false);
    }, []);

    return {
        messages,
        isResponding,
        responseLength,
        streamMode,
        onQuery,
        onInterrupt
    };
}

/**
 * Processes a single event from the main loop and updates UI state.
 */
function processMainLoopEvent(
    event: any,
    setMessages: (updater: (prev: any[]) => any[]) => void,
    setResponseLength: (updater: (prev: number) => number) => void,
    setStreamMode: (mode: any) => void
) {
    if (event.type === "stream_request_start") {
        setStreamMode("requesting");
        return;
    }

    if (event.type === "stream_event") {
        const { type } = event.event;
        if (type === "content_block_start") {
            const blockType = event.event.content_block.type;
            if (blockType === "thinking") setStreamMode("thinking");
            else if (blockType === "text") setStreamMode("responding");
            else if (blockType === "tool_use") setStreamMode("tool-input");
        } else if (type === "content_block_delta") {
            if (event.event.delta.text) {
                setResponseLength(prev => prev + event.event.delta.text.length);
            } else if (event.event.delta.partial_json) {
                setResponseLength(prev => prev + event.event.delta.partial_json.length);
            }
        }
        return;
    }

    if (event.type === "tombstone") {
        setMessages(prev => prev.filter(m => m.uuid !== event.message.uuid));
        return;
    }

    // Default: append message or tool result
    setMessages(prev => {
        // Check if message already exists
        const exists = prev.some(m => m.uuid === event.uuid || (m.type === "tool_result" && m.tool_use_id === event.tool_use_id));
        if (exists) return prev;
        return [...prev, event];
    });
}

