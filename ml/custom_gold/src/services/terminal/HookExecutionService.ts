
import { spawn, ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { log, logError } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { getGlobalState } from "../session/globalState.js";
import { createMetadataMessage } from "./MessageFactory.js";
import { formatDuration as getFormattedDuration } from "../../utils/shared/formatUtils.js";

// Logic from chunk_585 (Prompt/Agent Hooks), chunk_587 (Hook Execution/Bash), chunk_586 (Hook Output Parsing)

/**
 * Common input structure shared by all hooks.
 * Based on aF from chunk_585.ts
 */
import { z } from "zod";
import { callAnthropic } from "./anthropicStream.js";
import { mainLoop } from "./AgentMainLoop.js";
import { createUserMessage } from "./MessageFactory.js";


// Helper for template replacement (oI1)
function replaceTemplate(template: string, inputJson: string): string {
    // Determine replacement key (G) - typically 'input' or entire json
    // chunk_585 logic implies it replaces {{input}} with the JSON input
    // or does string replacement for other variables.
    // Simplified logic:
    return template.replace(/\{\{input\}\}/g, inputJson);
}

// Schemas from chunk_585.ts
const HookResultSchema = z.union([
    z.object({
        ok: z.literal(true)
    }),
    z.object({
        ok: z.literal(false),
        reason: z.string()
    })
]);

// Helper Message Extractor from chunk_585 (ub)
async function processUserPrompt(prompt: string, context: any) {
    // In chunk_585, 'ub' converts the prompt into messages and checks if query is needed.
    // For hooks, we generally just send the prompt as a user message.
    return {
        messages: [{ role: "user", content: prompt }],
        shouldQuery: true
    };
}

export function createBaseHookInput(permissionMode: string | undefined, sessionId: string) {
    return {
        session_id: sessionId,
        transcript_path: `/tmp/claude-transcript-${sessionId}.txt`, // Placeholder path logic
        cwd: process.cwd(),
        permission_mode: permissionMode ?? "unknown"
    };
}

/**
 * Parses the output from a hook.
 * Based on ZK9 from chunk_585.ts
 */
export function parseHookOutput(output: string) {
    const trimmed = output.trim();
    if (!trimmed.startsWith("{")) {
        return { plainText: trimmed };
    }
    try {
        const json = JSON.parse(trimmed);
        return { json, plainText: trimmed };
    } catch (err) {
        return { plainText: trimmed, parseError: err };
    }
}

/**
 * Get matching hooks from global state based on event and query.
 * Based on qM0 from chunk_587.ts
 */
export function getMatchingHooks(appState: any, hookEvent: string, hookInput: any) {
    // In a real implementation, this would query the Settings/Config
    // For now, we return empty or check appState.hooks
    // Retrieving matching hooks logic (OJ7) is complex involving settings merge
    // Simplified:
    const hooks = appState?.hooks?.[hookEvent] || [];
    // Filter by matcher if needed
    return hooks;
}

/**
 * Execute a Bash Hook.
 * Based on FD1 from chunk_587.ts
 */
async function executeBashHook(hook: any, hookEvent: string, hookName: string, inputJson: string, abortSignal: AbortSignal) {
    // Simplified Spawn logic
    try {
        const env = { ...process.env, CLAUDE_HOOK_INPUT: inputJson };
        const proc = spawn(hook.command, [], {
            shell: true,
            env,
            cwd: process.cwd()
        });

        let stdout = "";
        let stderr = "";

        if (abortSignal) {
            abortSignal.addEventListener('abort', () => {
                proc.kill();
            });
        }

        proc.stdout.on('data', (d) => stdout += d.toString());
        proc.stderr.on('data', (d) => stderr += d.toString());

        // Async hook detection logic (simplified)
        // If stdout contains { "async": true }, we should detach/background
        // For now, we wait for completion

        return new Promise<any>((resolve) => {
            proc.on('close', (code) => {
                resolve({
                    stdout,
                    stderr,
                    status: code,
                    aborted: abortSignal?.aborted
                });
            });
            proc.on('error', (err) => {
                resolve({
                    stdout,
                    stderr: `Spawn error: ${err.message}`,
                    status: 1
                });
            });
        });
    } catch (err: any) {
        return { stdout: "", stderr: err.message, status: 1 };
    }
}

/**
 * Execute a Prompt Hook.
 * Based on sW9 from chunk_585.ts
 */
async function executePromptHook(hook: any, hookName: string, hookEvent: string, inputJson: string, abortSignal: AbortSignal, context: any) {
    const prompt = replaceTemplate(hook.prompt, inputJson);
    log("hooks", `Executing prompt hook ${hookName} with prompt: ${prompt}`);

    try {
        const { messages, shouldQuery } = await processUserPrompt(prompt, context);
        if (!shouldQuery) {
            // Unlikely for prompt hook, but handle it
            return {
                outcome: "success",
                hook,
                message: createMetadataMessage({ type: "verbose", text: "Hook prompt required no query" })
            };
        }

        const systemPrompt = `You are evaluating a hook in Claude Code.

CRITICAL: You MUST return ONLY valid JSON with no other text, explanation, or commentary before or after the JSON. Do not include any markdown code blocks, thinking, or additional text.

Your response must be a single JSON object matching one of the following schemas:
1. If the condition is met, return: {"ok": true}
2. If the condition is not met, return: {"ok": false, "reason": "Reason for why it is not met"}

Return the JSON object directly with no preamble or explanation.`;

        // Extend context messages if valid
        const queryMessages = [...messages, { role: "assistant", content: "{" }];

        const response = await callAnthropic({
            messages: queryMessages,
            systemPrompt,
            maxThinkingTokens: 0,
            tools: context.options?.tools || [],
            signal: abortSignal,
            options: {
                // Mock options or pass from context
                model: hook.model || "claude-3-5-sonnet-20241022",
                querySource: "hook_prompt",
                mcpTools: [],
                agentId: context.agentId
            }
        });

        const content = response.message.content.filter((c: any) => c.type === "text").map((c: any) => c.text).join("");
        const fullJson = ("{" + content).trim();
        log("hooks", `Model response: ${fullJson}`);

        const result = parseHookOutput(fullJson);
        if (!result.json) {
            return {
                hook,
                outcome: "non_blocking_error",
                message: createMetadataMessage({ type: "verbose", text: `JSON validation failed: ${result.parseError}` })
            };
        }

        const validation = HookResultSchema.safeParse(result.json);
        if (!validation.success) {
            return {
                hook,
                outcome: "non_blocking_error",
                message: createMetadataMessage({ type: "verbose", text: `Schema validation failed: ${validation.error.message}` })
            };
        }

        if (!validation.data.ok) {
            return {
                hook,
                outcome: "blocking",
                blockingError: validation.data.reason,
                preventContinuation: true,
                stopReason: validation.data.reason
            };
        }

        return {
            hook,
            outcome: "success",
            message: createMetadataMessage({ type: "verbose", text: "Condition met" })
        };

    } catch (err: any) {
        if (abortSignal.aborted) return { hook, outcome: "cancelled" };
        logError("HookExecution", err);
        return {
            hook,
            outcome: "non_blocking_error",
            message: createMetadataMessage({ type: "verbose", text: `Error executing prompt hook: ${err.message}` })
        };
    }
}

/**
 * Execute an Agent Hook.
 * Based on AK9 from chunk_585.ts
 */
async function executeAgentHook(hook: any, hookName: string, hookEvent: string, inputJson: string, abortSignal: AbortSignal, context: any) {
    const prompt = replaceTemplate(hook.prompt, inputJson); // assuming template replacement applies
    log("hooks", `Executing agent hook ${hookName}`);

    try {
        const { messages, shouldQuery } = await processUserPrompt(prompt, context);
        if (!shouldQuery) return { hook, outcome: "success" };

        const systemPrompt = `You are verifying a stop condition in Claude Code. Your task is to verify that the agent completed the given plan.
Use the available tools to inspect the codebase and verify the condition.
Use as few steps as possible - be efficient and direct.

When done, return your result using the ${"structured_output"} tool with:
- ok: true if the condition is met
- ok: false with reason if the condition is not met`;

        const subAgentId = randomUUID();
        const mainLoopOptions = {
            messages,
            systemPrompt,
            canUseTool: async () => ({ behavior: "allow" }), // Auto-allow for checking
            toolUseContext: {
                ...context,
                agentId: subAgentId,
                abortController: new AbortController(), // Needs own controller
                getAppState: () => Promise.resolve(getGlobalState()),
            },
            querySource: "hook_agent",
            stopHookActive: true // Avoid recursive stop hooks
        };

        let resultData: any = null;
        let turnCount = 0;

        // Run Main Loop for Agent Hook
        for await (const event of mainLoop(mainLoopOptions)) {
            if (abortSignal.aborted) break;
            if (event.type === "assistant") {
                turnCount++;
                if (turnCount >= 50) {
                    log("hooks", "Agent hook hit max turns");
                    break;
                }
            }

            // Check for structured output result
            if (event.type === "tool_result" && event.is_error) {
                // handle error
            }

            // Simplification: In a real agent hook, we look for a specific output tool call.
            // Here we assume the agent uses 'structured_output' or similar, 
            // but identifying that from mainLoop events requires looking at `assistant` messages
            // containing tool_use.

            // In chunk_585, it checks for `e.type === "attachment" && e.attachment.type === "structured_output"`
            if (event.type === "attachment" && event.attachment?.type === "structured_output") {
                const parse = z.object({ ok: z.boolean(), reason: z.string().optional() }).safeParse(event.attachment.data);
                if (parse.success) {
                    resultData = parse.data;
                    break;
                }
            }
        }

        if (abortSignal.aborted) return { hook, outcome: "cancelled" };

        if (!resultData) {
            return {
                hook,
                outcome: "cancelled", // Treated as cancelled or failed
                message: createMetadataMessage({ type: "verbose", text: "Agent hook did not return structured output" })
            };
        }

        if (!resultData.ok) {
            return {
                hook,
                outcome: "blocking",
                blockingError: `Agent hook condition was not met: ${resultData.reason}`,
                preventContinuation: true
            };
        }

        return {
            hook,
            outcome: "success",
            message: createMetadataMessage({ type: "verbose", text: "Condition met" })
        };

    } catch (err: any) {
        if (abortSignal.aborted) return { hook, outcome: "cancelled" };
        logError("AgentHook", err);
        return {
            hook,
            outcome: "non_blocking_error",
            message: createMetadataMessage({ type: "verbose", text: `Error executing agent hook: ${err.message}` })
        };
    }
}


/**
 * Main generator to execute hooks.
 * Based on 'ms' from chunk_587.ts
 */
export async function* executeHooks(params: {
    hookInput: any,
    toolUseID: string,
    matchQuery?: string,
    signal?: AbortSignal,
    timeoutMs?: number,
    toolUseContext?: any
}) {
    const { hookInput, toolUseID, matchQuery, signal, timeoutMs = 60000, toolUseContext } = params;
    const hookEvent = hookInput.hook_event_name;
    const globalState = getGlobalState();

    // Check disableAllHooks key in settings
    // if (globalState.settings.disableAllHooks) return;

    const matchingHooks = getMatchingHooks(globalState, hookEvent, hookInput);
    if (!matchingHooks || matchingHooks.length === 0) return;

    logTelemetryEvent("tengu_run_hook", {
        hookName: `${hookEvent}:${matchQuery || ''}`,
        numCommands: matchingHooks.length
    });

    for (const hook of matchingHooks) {
        // Yield progress
        yield {
            type: "progress",
            message: `Running ${hookEvent} hook: ${hook.name || hook.command || "unknown"}`,
            toolUseID
        };

        // Determine type and execute matching runner
        let result: any = {};
        const inputJson = JSON.stringify(hookInput);

        // Abort signal for this specific hook
        const hookAbort = new AbortController();
        if (signal) {
            signal.addEventListener('abort', () => hookAbort.abort());
        }

        try {
            if (hook.type === "prompt") {
                result = await executePromptHook(hook, hook.name, hookEvent, inputJson, hookAbort.signal, toolUseContext);
            } else if (hook.type === "agent") {
                result = await executeAgentHook(hook, hook.name, hookEvent, inputJson, hookAbort.signal, toolUseContext);
            } else {
                // Default to bash command
                const execResult = await executeBashHook(hook, hookEvent, hook.name, inputJson, hookAbort.signal);
                const parsed = parseHookOutput(execResult.stdout);

                // Process result format (YK9 logic)
                if (parsed.json) {
                    // Map JSON to yield events (blocking error, messages, permission updates)
                    // ... YK9 logic maps fields like 'decision', 'permissionDecision', etc.
                    result = {
                        outcome: "success",
                        json: parsed.json,
                        stdout: execResult.stdout,
                        stderr: execResult.stderr,
                        status: execResult.status
                    };
                } else {
                    result = {
                        outcome: execResult.status === 0 ? "success" : "non_blocking_error",
                        stdout: execResult.stdout,
                        stderr: execResult.stderr
                    };
                }
            }
        } catch (err: any) {
            console.error("Hook execution error", err);
            result = { outcome: "non_blocking_error", error: err.message };
        }

        // Yield result events
        if (result.outcome === "blocking" || (result.json && result.json.decision === "block")) {
            yield {
                type: "error",
                error: result.blockingError || result.json?.reason || "Blocked by hook",
                blocking: true
            };
        }

        // Handle permission changes
        if (result.json?.permissionBehavior) {
            yield {
                type: "permission_update",
                behavior: result.json.permissionBehavior,
                reason: result.json.reason
            };
        }
    }
}


/**
 * Specific Event Runners (Wrappers around executeHooks)
 */

export async function* executePreToolHooks(toolName: string, toolInput: any, toolUseID: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"), // Needs real session ID
        hook_event_name: "PreToolUse",
        tool_name: toolName,
        tool_input: toolInput,
        tool_use_id: toolUseID
    };

    yield* executeHooks({
        hookInput,
        toolUseID,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function* executePostToolHooks(toolName: string, toolInput: any, toolResponse: any, toolUseID: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "PostToolUse",
        tool_name: toolName,
        tool_input: toolInput,
        tool_response: toolResponse,
        tool_use_id: toolUseID
    };

    yield* executeHooks({
        hookInput,
        toolUseID,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function* executePostToolUseFailureHooks(toolName: string, toolInput: any, error: any, toolUseID: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "PostToolUseFailure",
        tool_name: toolName,
        tool_input: toolInput,
        error: error instanceof Error ? error.message : String(error),
        tool_use_id: toolUseID
    };

    yield* executeHooks({
        hookInput,
        toolUseID,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function* executePermissionRequestHooks(toolName: string, toolInput: any, toolUseID: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "PermissionRequest",
        tool_name: toolName,
        tool_input: toolInput,
        tool_use_id: toolUseID
    };

    yield* executeHooks({
        hookInput,
        toolUseID,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function* executeSessionStartHooks(sessionSource: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "SessionStart",
        source: sessionSource
    };

    // Use a generic ID for session events
    const eventID = randomUUID();

    yield* executeHooks({
        hookInput,
        toolUseID: eventID,
        matchQuery: sessionSource,
        signal,
        toolUseContext: context
    });
}

export async function* executeSessionEndHooks(reason: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "SessionEnd",
        reason: reason
    };

    const eventID = randomUUID();

    yield* executeHooks({
        hookInput,
        toolUseID: eventID,
        matchQuery: reason,
        signal,
        toolUseContext: context
    });
}

export async function* executePreCompactHooks(trigger: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "PreCompact",
        trigger: trigger
    };

    const eventID = randomUUID();

    yield* executeHooks({
        hookInput,
        toolUseID: eventID,
        matchQuery: trigger,
        signal,
        toolUseContext: context
    });
}

export async function* executeNotificationHooks(type: string, data: any, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "Notification",
        notification_type: type,
        data: data
    };

    const eventID = randomUUID();

    yield* executeHooks({
        hookInput,
        toolUseID: eventID,
        matchQuery: type,
        signal,
        toolUseContext: context
    });
}

export async function* executeSubagentStartHooks(agentType: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "SubagentStart",
        agent_type: agentType
    };

    const eventID = randomUUID();

    yield* executeHooks({
        hookInput,
        toolUseID: eventID,
        matchQuery: agentType,
        signal,
        toolUseContext: context
    });
}

export async function* executeUserPromptSubmitHooks(prompt: string, context: any, signal?: AbortSignal) {
    const hookInput = {
        ...createBaseHookInput("unknown", "session-id"),
        hook_event_name: "UserPromptSubmit",
        prompt: prompt
    };

    const eventID = randomUUID();

    yield* executeHooks({
        hookInput,
        toolUseID: eventID,
        matchQuery: undefined,
        signal,
        toolUseContext: context
    });
}
