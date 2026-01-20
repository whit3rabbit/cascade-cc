
import { randomUUID } from "node:crypto";
import { log, logError } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import {
    normalizeMessages,
    createMetadataMessage,
    createToolResultMessage,
    createBannerMessage,
    createUserMessage
} from "../terminal/MessageFactory.js";
import { compactConversation, microCompactMessages } from "../history/ConversationHistoryManager.js";
import { executePreStopHooks } from "../hooks/HookExecutor.js";
import { getCompletionStream } from "../terminal/StreamingHandler.js";
import { ToolExecutionQueue } from "./ToolExecutionQueue.js";
import { getSessionId } from "../session/globalState.js";

const logger = log("loops");

export interface AutoCompactTracking {
    compacted: boolean;
    turnId: string;
    turnCounter: number;
}

export interface MainLoopOptions {
    messages: any[];
    systemPrompt: string;
    userContext: string;
    systemContext: string;
    canUseTool: any;
    toolUseContext: any;
    autoCompactTracking?: AutoCompactTracking;
    fallbackModel?: string;
    stopHookActive?: boolean;
    querySource?: string;
    maxOutputTokensOverride?: number;
}

/**
 * Core agent orchestration loop.
 * Based on chunk_457.ts (ew / mainLoop)
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

    const currentChainId = queryTracking.chainId;
    toolUseContext = { ...toolUseContext, queryTracking };

    // Standardize and Micro-compact (Vd)
    const { messages: normalizedMessages } = await microCompactMessages(messages, undefined, toolUseContext);
    let currentMessages = normalizedMessages;

    // Check for auto-compaction (CT2)
    const { compactionResult } = await compactConversation(currentMessages, toolUseContext, querySource);
    let tracking = autoCompactTracking;

    if (compactionResult) {
        logTelemetryEvent("tengu_auto_compact_succeeded", {
            originalMessageCount: messages.length,
            compactedMessageCount: compactionResult.summaryMessages.length + (compactionResult.attachments?.length ?? 0) + (compactionResult.hookResults?.length ?? 0),
            preCompactTokenCount: compactionResult.preCompactTokenCount,
            postCompactTokenCount: compactionResult.postCompactTokenCount,
            queryChainId: currentChainId,
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

    const assistantMessages: any[] = [];
    const toolResults: any[] = [];

    // Check if we should use streaming tool execution (chunk_457.ts:367)
    // We'll assume the feature flag "tengu_streaming_tool_execution2" is true
    const executionQueue = new ToolExecutionQueue(toolUseContext.options.tools, canUseTool, toolUseContext);

    const appState = await toolUseContext.getAppState();
    const mode = appState.toolPermissionContext.mode;

    // Determine model based on mode and context size (D1A)
    const model = getModel({
        permissionMode: mode,
        mainLoopModel: toolUseContext.options.mainLoopModel || appState.model,
        messages: currentMessages,
        fallbackModel
    });

    const stream = getCompletionStream({
        messages: currentMessages,
        systemPrompt: formatSystemPrompt(systemPrompt, userContext, systemContext),
        maxThinkingTokens: toolUseContext.options.maxThinkingTokens,
        tools: toolUseContext.options.tools,
        signal: toolUseContext.abortController.signal,
        options: {
            model,
            querySource,
            agentId: toolUseContext.agentId,
            queryTracking,
            maxOutputTokensOverride,
            mcpTools: appState.mcp?.tools ?? []
        }
    });

    try {
        for await (const chunk of stream) {
            yield chunk;

            if (chunk.type === "assistant") {
                assistantMessages.push(chunk);
                const toolUses = chunk.message.content.filter((c: any) => c.type === "tool_use");
                for (const toolUse of toolUses) {
                    logTelemetryEvent("tengu_tool_use", {
                        client_request_id: chunk.requestId,
                        tool_use_id: toolUse.id,
                        tool_name: toolUse.name,
                        tool_use_input: JSON.stringify(toolUse.input),
                        is_mcp: toolUse.name.startsWith("mcp__"),
                        query_chain_id: queryTracking.chainId,
                        query_depth: queryTracking.depth
                    });
                    executionQueue.addTool(toolUse, chunk);
                }
            }

            for (const result of Array.from(executionQueue.getCompletedResults())) {
                if (result.message) {
                    yield result.message;
                    if (result.message.type === "user") {
                        toolResults.push(result.message);
                    }
                }
            }
        }
    } catch (err: any) {
        logError("loops", err, "Main loop error during streaming");
        logTelemetryEvent("tengu_query_error", {
            assistantMessages: assistantMessages.length,
            queryChainId: currentChainId,
            queryDepth: queryTracking.depth
        });

        // Create error results for any unfinished tools (kF0)
        for (const msg of Array.from(createToolResultErrorMessages(assistantMessages, err instanceof Error ? err.message : String(err)))) {
            yield msg;
        }
        throw err;
    }

    if (toolUseContext.abortController.signal.aborted) {
        for await (const res of executionQueue.getRemainingResults()) {
            if (res.message) yield res.message;
        }
        // Yield interrupted error messages if not yielded by queue
        for (const msg of Array.from(createToolResultErrorMessages(assistantMessages, "Interrupted by user"))) {
            yield msg;
        }
        return;
    }

    const allToolUses = assistantMessages.flatMap(m => m.message.content.filter((c: any) => c.type === "tool_use"));

    if (assistantMessages.length === 0 || allToolUses.length === 0) {
        // End of turn hooks (ag5 / executePreStopHooks)
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

            // Generate prompt suggestions (Wv2)
            generatePromptSuggestion({
                messages: [...currentMessages, ...assistantMessages, ...toolResults],
                toolUseContext,
                querySource
            }).catch(err => logError("loops", err, "Suggestion generation failed"));
        }
        return;
    }

    // Process remaining tools in queue
    let nextToolUseContext = toolUseContext;
    for await (const res of executionQueue.getRemainingResults()) {
        if (res.message) {
            yield res.message;
            if (res.message.type === "user") {
                toolResults.push(res.message);
            }
            if (res.message.type === "attachment" && res.message.attachment?.type === "hook_stopped_continuation") {
                return; // Stop loop
            }
        }
    }
    nextToolUseContext = { ...executionQueue.getUpdatedContext(), queryTracking };

    if (tracking?.compacted) {
        tracking.turnCounter++;
    }

    // Handle steering attachments / queued commands (FHA / eY2)
    const stateWithQueue = await nextToolUseContext.getAppState();
    const queuedCommands = stateWithQueue.queuedCommands || [];
    const steeringAttachments: any[] = [];

    if (queuedCommands.length > 0) {
        logger.info(`MainLoop: Processing ${queuedCommands.length} queued commands`);
        for (const cmd of queuedCommands) {
            steeringAttachments.push(createUserMessage(cmd));
        }
        // Clear queue
        await nextToolUseContext.setAppState((s: any) => ({ ...s, queuedCommands: [] }));
    }

    if (nextToolUseContext.pendingSteeringAttachments) {
        steeringAttachments.push(...nextToolUseContext.pendingSteeringAttachments);
    }

    // Recurse
    yield* mainLoop({
        messages: [...currentMessages, ...assistantMessages, ...toolResults, ...steeringAttachments],
        systemPrompt,
        userContext,
        systemContext,
        canUseTool,
        toolUseContext: { ...nextToolUseContext, pendingSteeringAttachments: steeringAttachments.length > 0 ? steeringAttachments : undefined },
        autoCompactTracking: tracking,
        fallbackModel,
        stopHookActive,
        querySource
    });
}

/**
 * Logic to decide which model to use. (D1A)
 */
function getModel(options: {
    permissionMode: string,
    mainLoopModel: string,
    messages: any[],
    fallbackModel?: string
}): string {
    const { mainLoopModel, messages, fallbackModel } = options;

    // Check for "claude-3-5-sonnet" preference but check message size
    // In chunk_457, they sometimes switch models if context is very large
    // For now, respect the main loop model choice
    return mainLoopModel || fallbackModel || "claude-3-5-sonnet-20241022";
}

/**
 * Formats the final system prompt with available context. (dy2)
 */
function formatSystemPrompt(prompt: string, user: string, system: string): string {
    let finalPrompt = prompt;
    if (user) {
        finalPrompt += `\n\n<user_context>\n${user}\n</user_context>`;
    }
    if (system) {
        finalPrompt += `\n\n<system_context>\n${system}\n</system_context>`;
    }
    return finalPrompt;
}

/**
 * Creates error results for tools that were called but never finished.
 * Based on chunk_457.ts (kF0)
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
                toolUseResult: error
            });
        }
    }
}

let suggestionAbortController: AbortController | null = null;

const SUGGESTION_PROMPT_DEFAULT = `You are now a prompt suggestion generator. The conversation above is context - your job is to suggest what Claude could help with next.

Based on the conversation, suggest the user's next prompt. Short casual input, 3-8 words. Read the moment - what's the natural next step?

Be specific when you can. Even if the task seems done, think about natural follow-ups. Say "done" only if the work is truly complete.

Reply with ONLY the suggestion text, no quotes, no explanation, no markdown.`;

const SUGGESTION_PROMPT_INTENT = `[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.]

FIRST: Look at the user's recent messages and original request.

Your job is to predict what THEY would type - not what you think they should do.

THE TEST: Would they think "I was just about to type that"?

EXAMPLES:
User asked "fix the bug and run tests", bug is fixed → "run the tests"
After code written → "try it out"
Claude offers options → suggest the one the user would likely pick, based on conversation
Claude asks to continue → "yes" or "go ahead"
Task complete, obvious follow-up → "commit this" or "push it"
After error or misunderstanding → silence (let them assess/correct)

Be specific: "run the tests" beats "continue".

NEVER SUGGEST:
- Evaluative ("looks good", "thanks")
- Questions ("what about...?")
- Claude-voice ("Let me...", "I'll...", "Here's...")
- New ideas they didn't ask about
- Multiple sentences

Stay silent if the next step isn't obvious from what the user said.

Format: 2-8 words, match the user's style. Or nothing.

Reply with ONLY the suggestion, no quotes or explanation.`;

/**
 * Generates prompt suggestions in the background.
 * Based on chunk_457.ts (Wv2)
 */
export async function generatePromptSuggestion(options: {
    messages: any[],
    toolUseContext: any,
    querySource?: string
}) {
    if (options.querySource !== "repl_main_thread") return;

    const state = await options.toolUseContext.getAppState();
    if (!state.promptSuggestionEnabled) return;

    if (state.pendingWorkerRequest || state.pendingSandboxRequest) return;
    if (state.elicitation?.queue?.length > 0) return;
    if (state.toolPermissionContext.mode === "plan") return;

    // Early conversation check
    if (options.messages.filter(m => m.type === "assistant").length < 2) return;

    const lastMessage = options.messages[options.messages.length - 1];
    if (lastMessage?.isApiErrorMessage) return;

    // Abort previous
    if (suggestionAbortController) {
        suggestionAbortController.abort();
    }

    suggestionAbortController = new AbortController();
    const signal = suggestionAbortController.signal;

    try {
        const { suggestion, generationRequestId } = await callPromptSuggestionModel(options, signal);
        if (suggestion && !shouldSuppressSuggestion(suggestion)) {
            options.toolUseContext.setAppState((s: any) => ({
                ...s,
                promptSuggestion: {
                    text: suggestion,
                    promptId: "suggestion_generator",
                    shownAt: 0,
                    acceptedAt: 0,
                    generationRequestId
                }
            }));
        }
    } catch (err) {
        if (!(err instanceof Error && err.name === "AbortError")) {
            logError("loops", err, "Failed to generate prompt suggestion");
        }
    } finally {
        if (suggestionAbortController?.signal === signal) {
            suggestionAbortController = null;
        }
    }
}

async function callPromptSuggestionModel(options: any, signal: AbortSignal): Promise<{ suggestion: string | null, generationRequestId: string | null }> {
    const { messages, toolUseContext } = options;

    // Deobfuscated from ug5 in chunk_457:64
    const prompt = SUGGESTION_PROMPT_INTENT; // Using intent mode as default premium feel

    const results: any[] = [];
    const loop = mainLoop({
        messages: [...messages, { type: "user", content: prompt, isMeta: true }],
        systemPrompt: "You are a helpful assistant.",
        userContext: "",
        systemContext: "",
        canUseTool: async () => ({
            behavior: "deny",
            message: "No tools needed for suggestion",
            decisionReason: { type: "other", reason: "suggestion only" }
        }),
        toolUseContext: {
            ...toolUseContext,
            abortController: new AbortController(), // Separate controller for the fork
            setAppState: () => { }, // Don't allow state changes
            setMessages: () => { },
            setInProgressToolUseIDs: () => { }
        },
        querySource: "prompt_suggestion",
        stopHookActive: true // Avoid infinite loops/hooks
    });

    try {
        for await (const chunk of loop) {
            results.push(chunk);
        }
    } catch (err) {
        return { suggestion: null, generationRequestId: null };
    }

    const assistantMsg = results.find(m => m.type === "assistant");
    const requestId = assistantMsg?.requestId ?? null;

    const textPart = assistantMsg?.message?.content?.find((c: any) => c.type === "text");
    const suggestion = textPart?.text?.trim() || null;

    return { suggestion, generationRequestId: requestId };
}

function shouldSuppressSuggestion(text: string): boolean {
    if (!text) return true;
    const low = text.toLowerCase();
    const words = text.trim().split(/\s+/).length;

    if (low === "done") return true;
    if (words < 2) return true;
    if (words > 8) return true;
    if (text.length >= 100) return true;

    // API error messages
    if (low.startsWith("api error:") || low.startsWith("prompt is too long") ||
        low.startsWith("request timed out") || low.startsWith("invalid api key") ||
        low.startsWith("image was too large")) return true;

    if (/[.!?]\s+[A-Z]/.test(text)) return true; // multiple sentences
    if (/[\n*]|\*\*/.test(text)) return true; // has formatting

    // Evaluative
    if (/thanks|thank you|looks good|sounds good|that works|that worked|that's all|nice|great|perfect|makes sense|awesome|excellent/.test(low)) return true;

    // Claude voice
    if (/^(let me|i'll|i've|i'm|i can|i would|i think|i notice|here's|here is|here are|that's|this is|this will|you can|you should|you could|sure,|of course|certainly)/i.test(text)) return true;

    return false;
}
