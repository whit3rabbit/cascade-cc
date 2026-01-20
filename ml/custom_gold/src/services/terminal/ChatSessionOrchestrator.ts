
import { randomUUID } from "node:crypto";
import { useEffect, useRef } from "react";
import { useInput } from "ink";
import { mainLoop } from "../loops/MainLoop.js";
import {
    createUserMessage,
    createAssistantMessage,
    createMetadataMessage
} from "./MessageFactory.js";
import { log, logError } from "../logger/loggerService.js";
import { getSessionId } from "../session/globalState.js";
import { ToolExecutionQueue } from "../loops/ToolExecutionQueue.js";

const logger = log("ChatOrchestrator");

/**
 * Handles an orphaned permission result.
 * Based on chunk_516.ts (D07)
 */
async function* handleOrphanedPermission(
    orphanedPermission: any,
    tools: any[],
    mutableMessages: any[],
    toolUseContext: any,
    sessionId: string
): AsyncGenerator<any> {
    const { permissionResult, assistantMessage } = orphanedPermission;
    const toolUseId = permissionResult.toolUseID;

    if (!toolUseId) return;

    const content = assistantMessage.message.content;
    let toolUse: any;
    if (Array.isArray(content)) {
        toolUse = content.find((c: any) => c.type === "tool_use" && c.id === toolUseId);
    }

    if (!toolUse) return;

    // Verify tool still exists
    if (!tools.find(t => t.name === toolUse.name)) return;

    const updatedToolUse = {
        ...toolUse,
        input: permissionResult.behavior === "allow" ? permissionResult.updatedInput : toolUse.input
    };

    // Callback that immediately returns the captured permission
    const capturedPermissionSource = async () => ({
        ...permissionResult,
        decisionReason: {
            type: "mode",
            mode: "default"
        }
    });

    // Re-inject the assistant message that triggered the permission
    mutableMessages.push(assistantMessage);
    yield {
        ...assistantMessage,
        session_id: sessionId,
        parent_tool_use_id: null
    };

    const executionQueue = new ToolExecutionQueue(tools, capturedPermissionSource, toolUseContext);
    executionQueue.addTool(updatedToolUse, assistantMessage);

    for await (const res of executionQueue.getRemainingResults()) {
        if (res.message) {
            mutableMessages.push(res.message);
            yield {
                ...res.message,
                session_id: sessionId,
                parent_tool_use_id: null
            };
        }
    }
}

/**
 * Orchestrates a chat session, managing the main loop, state updates, and event yielding.
 * Deobfuscated/Adapted from `qe2` in `chunk_516.ts`.
 */
export async function* runChatSession(options: any): AsyncGenerator<any> {
    const {
        commands,
        prompt,
        promptUuid,
        cwd,
        tools,
        mcpClients,
        verbose = false,
        maxThinkingTokens,
        maxTurns,
        maxBudgetUsd,
        canUseTool, // Permission callback
        mutableMessages = [],
        customSystemPrompt,
        appendSystemPrompt,
        userSpecifiedModel,
        fallbackModel,
        sdkBetas,
        getAppState,
        setAppState,
        abortController,
        replayUserMessages = false,
        agents = [],
        setSDKStatus,
        orphanedPermission
    } = options;

    const startTime = Date.now();
    const sessionId = getSessionId();

    // Permission tracking
    const permissionDenials: any[] = [];
    const wrappedCanUseTool = async (tool: any, input: any, context: any) => {
        const result = await canUseTool(tool, input, context);
        if (result.behavior !== "allow") {
            permissionDenials.push({
                tool_name: tool.name,
                tool_use_id: context.id,
                tool_input: input
            });
        }
        return result;
    };

    const appState = await getAppState();
    const currentModel = userSpecifiedModel || appState.model;

    // Tool Context for MainLoop
    const toolUseContext = {
        options: {
            commands,
            tools,
            verbose,
            mainLoopModel: currentModel,
            maxThinkingTokens: maxThinkingTokens ?? 0,
            mcpClients,
            customSystemPrompt,
            appendSystemPrompt,
            agentDefinitions: { activeAgents: agents, allAgents: [] },
            maxBudgetUsd,
            sdkBetas,
            isNonInteractiveSession: true
        },
        getAppState,
        setAppState,
        abortController: abortController || new AbortController(),
        setInProgressToolUseIDs: () => { },
        agentId: appState.agent,
        setSDKStatus
    };

    // Initial System Yield
    yield {
        type: "system",
        subtype: "init",
        cwd,
        session_id: sessionId,
        tools: tools.map((t: any) => t.name),
        mcp_servers: mcpClients.map((c: any) => ({ name: c.name, status: c.status })),
        model: currentModel,
        permissionMode: appState.toolPermissionContext.mode,
        slash_commands: commands.map((c: any) => c.name),
        uuid: randomUUID()
    };

    // Step 1: Handle orphaned permissions if present (D07)
    if (orphanedPermission) {
        yield* handleOrphanedPermission(orphanedPermission, tools, mutableMessages, toolUseContext, sessionId);
    }

    // Step 2: Handle initial prompt
    if (prompt) {
        const userMsg = createUserMessage(prompt, { uuid: promptUuid });
        mutableMessages.push(userMsg);
        yield {
            ...userMsg,
            session_id: sessionId,
            parent_tool_use_id: null
        };
    }

    // Main Loop Execution
    let turnCount = 1; // Starting turn count

    try {
        const mainLoopIterator = mainLoop({
            messages: mutableMessages,
            systemPrompt: customSystemPrompt || "You are Claude Code.",
            userContext: "",
            systemContext: "",
            canUseTool: wrappedCanUseTool,
            toolUseContext,
            fallbackModel,
            querySource: "sdk"
        });

        for await (const event of mainLoopIterator) {
            // Update local messages state
            if (event.type === "assistant" || event.type === "user" || (event.type === "system" && event.subtype === "compact_boundary")) {
                mutableMessages.push(event);
            }

            if (event.type === "user") turnCount++;

            // Pass through event
            yield {
                ...event,
                session_id: sessionId,
                parent_tool_use_id: event.parent_tool_use_id ?? null
            };

            // Check max turns
            if (maxTurns && turnCount >= maxTurns) {
                yield {
                    type: "result",
                    subtype: "error_max_turns",
                    is_error: false,
                    num_turns: turnCount,
                    session_id: sessionId,
                    permission_denials: permissionDenials,
                    uuid: randomUUID()
                };
                return;
            }

            // Check budget logic
            const latestState = await getAppState();
            if (maxBudgetUsd && latestState.totalCostUsd >= maxBudgetUsd) {
                yield {
                    type: "result",
                    subtype: "error_max_budget_usd",
                    is_error: false,
                    num_turns: turnCount,
                    session_id: sessionId,
                    permission_denials: permissionDenials,
                    uuid: randomUUID()
                };
                return;
            }
        }
    } catch (error: any) {
        logError("ChatOrchestrator", error);
        yield {
            type: "result",
            subtype: "error_during_execution",
            is_error: true,
            errors: [error.message || String(error)],
            session_id: sessionId,
            num_turns: turnCount,
            uuid: randomUUID()
        };
        return;
    }

    // Success Result
    yield {
        type: "result",
        subtype: "success",
        is_error: false,
        duration_ms: Date.now() - startTime,
        num_turns: turnCount,
        session_id: sessionId,
        permission_denials: permissionDenials,
        uuid: randomUUID(),
        total_cost_usd: (await getAppState()).totalCostUsd
    };
}

/**
 * Hook for managing IDE selection state via MCP notifications.
 * Based on Ue2 in chunk_516.ts.
 */
export function useIdeSelection(mcpClient: any, onSelectionChange: (selection: any) => void) {
    const isMounted = useRef(false);
    const currentClient = useRef(null);

    useEffect(() => {
        const client = mcpClient;

        if (currentClient.current !== client) {
            isMounted.current = false;
            currentClient.current = client || null;
            onSelectionChange({
                lineCount: 0,
                lineStart: undefined,
                text: undefined,
                filePath: undefined
            });
        }

        if (isMounted.current || !client) return;

        const handleNotification = (params: any) => {
            if (params.selection?.start && params.selection?.end) {
                const { start, end } = params.selection;
                let lineCount = end.line - start.line + 1;
                if (end.character === 0) lineCount--;

                onSelectionChange({
                    lineCount,
                    lineStart: start.line,
                    text: params.text,
                    filePath: params.filePath
                });
            } else if (params.text !== undefined) {
                onSelectionChange({
                    selection: null,
                    text: params.text,
                    filePath: params.filePath
                });
            }
        };

        if (client && typeof client.setNotificationHandler === 'function') {
            client.setNotificationHandler("selection_changed", handleNotification);
        }

        isMounted.current = true;
    }, [mcpClient, onSelectionChange]);
}

/**
 * Hook for managing transcript shortcuts (Ctrl+O, etc.)
 * Based on Ce2 in chunk_516.ts.
 */
export function useTranscriptShortcuts(handlers: {
    onToggleTranscript: () => void,
    onToggleInput: () => void,
    onCancel: () => void,
    mode: "transcript" | "prompt"
}) {
    useInput((input, key) => {
        if (key.ctrl && input === 'o') {
            handlers.onToggleTranscript();
        }
        if (key.ctrl && input === 'e' && handlers.mode === "transcript") {
            handlers.onToggleInput();
        }
        if ((key.ctrl && input === 'c') || (key.escape && handlers.mode === "transcript")) {
            handlers.onCancel();
        }
    });
}
