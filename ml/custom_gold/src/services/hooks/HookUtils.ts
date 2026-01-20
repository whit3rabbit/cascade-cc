
import { z } from "zod";
import { log } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { createAttachment } from "../history/ConversationHistoryManager.js";
import {
    executePreToolHooks,
    executePostToolHooks,
    executePostToolFailureHooks
} from "../terminal/LifecycleService.js";

const logger = log("hooks");

/**
 * Normalizes path for error reporting. (Ev2)
 */
function formatPath(path: (string | number)[]): string {
    if (path.length === 0) return "";
    return path.reduce((acc, curr, idx) => {
        if (typeof curr === "number") return `${acc}[${curr}]`;
        return idx === 0 ? String(curr) : `${acc}.${curr}`;
    }, "") as string;
}

/**
 * Formats Zod validation errors for tool input. (Zu5)
 */
export function formatValidationErrors(toolName: string, error: z.ZodError): string {
    const requiredMissing = error.issues
        .filter(e => e.code === "invalid_type" && (e as any).received === "undefined" && e.message === "Required")
        .map(e => formatPath(e.path as (string | number)[]));

    const unrecognizedKeys = error.issues
        .filter(e => e.code === "unrecognized_keys")
        .flatMap(e => (e as any).keys);

    const typeMismatches = error.issues
        .filter(e => e.code === "invalid_type" && "received" in e && (e as any).received !== "undefined" && e.message !== "Required")
        .map(e => ({
            param: formatPath(e.path as (string | number)[]),
            expected: (e as any).expected,
            received: (e as any).received
        }));

    const issues: string[] = [];
    if (requiredMissing.length > 0) {
        issues.push(...requiredMissing.map(p => `The required parameter \`${p}\` is missing`));
    }
    if (unrecognizedKeys.length > 0) {
        issues.push(...unrecognizedKeys.map(k => `An unexpected parameter \`${k}\` was provided`));
    }
    if (typeMismatches.length > 0) {
        issues.push(...typeMismatches.map(({ param, expected, received }) =>
            `The parameter \`${param}\` type is expected as \`${expected}\` but provided as \`${received}\``));
    }

    if (issues.length > 0) {
        return `${toolName} failed due to the following ${issues.length > 1 ? "issues" : "issue"}:\n${issues.join("\n")}`;
    }
    return error.message;
}

/**
 * Formats hook execution errors with truncation. (NX1/hF0)
 */
export function formatHookError(error: any): string {
    if (!error) return "Command failed with no output";

    let combined = "";
    if (error.code !== undefined || error.interrupted !== undefined) {
        // ShellError-like structure
        const parts = [
            error.code !== undefined ? `Exit code ${error.code}` : "",
            error.interrupted ? "Interrupted" : "",
            error.stderr || "",
            error.stdout || ""
        ];
        combined = parts.filter(Boolean).join("\n").trim();
    } else if (error instanceof Error) {
        const parts = [error.message];
        if ("stderr" in error && typeof (error as any).stderr === "string") parts.push((error as any).stderr);
        if ("stdout" in error && typeof (error as any).stdout === "string") parts.push((error as any).stdout);
        combined = parts.filter(Boolean).join("\n").trim();
    } else {
        combined = String(error);
    }

    if (!combined) return "Command failed with no output";
    if (combined.length <= 10000) return combined;

    const limit = 5000;
    return `${combined.slice(0, limit)}\n\n... [${combined.length - limit * 2} characters truncated] ...\n\n${combined.slice(-limit)}`;
}

/**
 * Pre-tool execution hooks logic. (Gu5)
 */
export async function* preToolExecutionHooks(
    context: any,
    tool: any,
    input: any,
    toolUseID: string,
    messageID: string,
    requestId?: string,
    mcpServerType?: string
) {
    const startTime = Date.now();
    try {
        const appState = await context.getAppState();
        const hooks = executePreToolHooks(
            tool.name,
            toolUseID,
            input,
            context,
            appState.toolPermissionContext.mode,
            context.abortController.signal
        );

        for await (const res of hooks) {
            if (res.message) {
                yield { type: "message", message: { message: res.message } };
            }

            if (res.blockingError) {
                const msg = `PreToolUse:${tool.name} blocking error: ${res.blockingError.blockingError}`;
                yield {
                    type: "hookPermissionResult",
                    hookPermissionResult: {
                        behavior: "deny",
                        message: msg,
                        decisionReason: {
                            type: "hook",
                            hookName: `PreToolUse:${tool.name}`,
                            reason: msg
                        }
                    }
                };
            }

            if (res.preventContinuation) {
                yield { type: "preventContinuation", shouldPreventContinuation: true };
                if (res.stopReason) yield { type: "stopReason", stopReason: res.stopReason };
            }

            if ((res as any).hookSpecificOutput?.hookEventName === "PreToolUse") {
                const out = (res as any).hookSpecificOutput;
                const decisionReason = {
                    type: "hook",
                    hookName: `PreToolUse:${tool.name}`,
                    reason: out.permissionDecisionReason
                };

                if (out.permissionDecision === "allow") {
                    yield {
                        type: "hookPermissionResult",
                        hookPermissionResult: {
                            behavior: "allow",
                            updatedInput: out.updatedInput || input,
                            decisionReason
                        }
                    };
                } else if (out.permissionDecision) {
                    yield {
                        type: "hookPermissionResult",
                        hookPermissionResult: {
                            behavior: out.permissionDecision,
                            message: out.permissionDecisionReason || `Hook PreToolUse:${tool.name} ${out.permissionDecision}ed this tool`,
                            decisionReason
                        }
                    };
                }
            }

            if (context.abortController.signal.aborted) {
                logTelemetryEvent("tengu_pre_tool_hooks_cancelled", {
                    toolName: tool.name,
                    queryChainId: context.queryTracking?.chainId,
                    queryDepth: context.queryTracking?.depth
                });
                yield {
                    type: "message",
                    message: {
                        message: createAttachment({
                            type: "hook_cancelled",
                            hookName: `PreToolUse:${tool.name}`,
                            toolUseID,
                            hookEvent: "PreToolUse"
                        })
                    }
                };
                yield { type: "stop" };
                return;
            }
        }
    } catch (err) {
        const duration = Date.now() - startTime;
        logTelemetryEvent("tengu_pre_tool_hook_error", {
            messageID,
            toolName: tool.name,
            duration,
            queryChainId: context.queryTracking?.chainId,
            queryDepth: context.queryTracking?.depth
        });
        yield {
            type: "message",
            message: {
                message: createAttachment({
                    type: "hook_error_during_execution",
                    content: formatHookError(err),
                    hookName: `PreToolUse:${tool.name}`,
                    toolUseID,
                    hookEvent: "PreToolUse"
                })
            }
        };
        yield { type: "stop" };
    }
}

/**
 * Post-tool execution hooks logic. (Qu5)
 */
export async function* postToolExecutionHooks(
    context: any,
    tool: any,
    toolUseID: string,
    messageID: string,
    toolOutput: any,
    requestId?: string,
    mcpServerType?: string
) {
    const startTime = Date.now();
    try {
        const appState = await context.getAppState();
        const mode = appState.toolPermissionContext.mode;
        let finalOutput = toolOutput;

        const hooks = executePostToolHooks(
            tool.name,
            toolUseID,
            toolOutput,
            context,
            mode,
            context.abortController.signal
        );

        for await (const res of hooks) {
            if (res.message?.type === "attachment" && res.message.attachment.type === "hook_cancelled") {
                logTelemetryEvent("tengu_post_tool_hooks_cancelled", {
                    toolName: tool.name,
                    queryChainId: context.queryTracking?.chainId,
                    queryDepth: context.queryTracking?.depth
                });
                yield {
                    message: createAttachment({
                        type: "hook_cancelled",
                        hookName: `PostToolUse:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUse"
                    })
                };
                continue;
            }

            if (res.message) yield { message: res.message };

            if (res.blockingError) {
                yield {
                    message: createAttachment({
                        type: "hook_blocking_error",
                        hookName: `PostToolUse:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUse",
                        blockingError: res.blockingError
                    })
                };
            }

            if (res.preventContinuation) {
                yield {
                    message: createAttachment({
                        type: "hook_stopped_continuation",
                        message: res.stopReason || "Execution stopped by PostToolUse hook",
                        hookName: `PostToolUse:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUse"
                    })
                };
                return;
            }

            if (res.additionalContexts && (res as any).additionalContexts.length > 0) {
                yield {
                    message: createAttachment({
                        type: "hook_additional_context",
                        content: res.additionalContexts,
                        hookName: `PostToolUse:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUse"
                    })
                };
            }

            if (res.updatedMCPToolOutput && tool.name.startsWith("mcp__")) {
                finalOutput = res.updatedMCPToolOutput;
                yield { updatedMCPToolOutput: finalOutput };
            }
        }
    } catch (err) {
        const duration = Date.now() - startTime;
        logTelemetryEvent("tengu_post_tool_hook_error", {
            messageID,
            toolName: tool.name,
            duration,
            queryChainId: context.queryTracking?.chainId,
            queryDepth: context.queryTracking?.depth
        });
        yield {
            message: createAttachment({
                type: "hook_error_during_execution",
                content: formatHookError(err),
                hookName: `PostToolUse:${tool.name}`,
                toolUseID,
                hookEvent: "PostToolUse"
            })
        };
    }
}

/**
 * Post-tool failure hooks logic. (Bu5)
 */
export async function* postToolFailureHooks(
    context: any,
    tool: any,
    toolUseID: string,
    messageID: string,
    input: any,
    error: any,
    isInterrupt: boolean,
    requestId?: string,
    mcpServerType?: string
) {
    const startTime = Date.now();
    try {
        const appState = await context.getAppState();
        const mode = appState.toolPermissionContext.mode;

        const hooks = executePostToolFailureHooks(
            tool.name,
            toolUseID,
            input,
            error,
            context,
            mode,
            context.abortController.signal,
            isInterrupt
        );

        for await (const res of hooks) {
            if (res.message?.type === "attachment" && res.message.attachment.type === "hook_cancelled") {
                logTelemetryEvent("tengu_post_tool_failure_hooks_cancelled", {
                    toolName: tool.name,
                    queryChainId: context.queryTracking?.chainId,
                    queryDepth: context.queryTracking?.depth
                });
                yield {
                    message: createAttachment({
                        type: "hook_cancelled",
                        hookName: `PostToolUseFailure:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUseFailure"
                    })
                };
                continue;
            }

            if (res.message) yield { message: res.message };

            if (res.blockingError) {
                yield {
                    message: createAttachment({
                        type: "hook_blocking_error",
                        hookName: `PostToolUseFailure:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUseFailure",
                        blockingError: res.blockingError
                    })
                };
            }

            if (res.additionalContexts && (res as any).additionalContexts.length > 0) {
                yield {
                    message: createAttachment({
                        type: "hook_additional_context",
                        content: res.additionalContexts,
                        hookName: `PostToolUseFailure:${tool.name}`,
                        toolUseID,
                        hookEvent: "PostToolUseFailure"
                    })
                };
            }
        }
    } catch (err) {
        const duration = Date.now() - startTime;
        logTelemetryEvent("tengu_post_tool_failure_hook_error", {
            messageID,
            toolName: tool.name,
            duration,
            queryChainId: context.queryTracking?.chainId,
            queryDepth: context.queryTracking?.depth
        });
        yield {
            message: createAttachment({
                type: "hook_error_during_execution",
                content: formatHookError(err),
                hookName: `PostToolUseFailure:${tool.name}`,
                toolUseID,
                hookEvent: "PostToolUseFailure"
            })
        };
    }
}

/**
 * Thinking logic. ($d, Wu5, cyA, OX1)
 */
export const THINKING_LEVELS = {
    NONE: 0,
    ULTRATHINK: 31999
};

export function calculateThinkingTokens(messages: any[], defaultTokens: number): number {
    if (process.env.MAX_THINKING_TOKENS) {
        const val = parseInt(process.env.MAX_THINKING_TOKENS, 10);
        return isNaN(val) ? 0 : val;
    }

    const thinkingMessages = messages.filter(m => m.type === "user" && !m.isMeta);
    if (thinkingMessages.length === 0) return defaultTokens || 0;

    const tokens = thinkingMessages.map(m => {
        if (m.thinkingMetadata) {
            const { level, disabled } = m.thinkingMetadata;
            if (disabled) return 0;
            return level === "high" ? THINKING_LEVELS.ULTRATHINK : 0;
        }
        const text = typeof m.message.content === "string" ? m.message.content :
            m.message.content.map((c: any) => c.text || "").join("");
        return /\bultrathink\b/i.test(text) ? THINKING_LEVELS.ULTRATHINK : 0;
    });

    return Math.max(...tokens, defaultTokens || 0);
}

export function isThinkingEnabled(): boolean {
    if (process.env.MAX_THINKING_TOKENS) return parseInt(process.env.MAX_THINKING_TOKENS, 10) > 0;

    // Logic from OX1/Vu5
    // return true for specific models or settings
    return true;
}

/**
 * Sub-agent context creation. (qv2)
 */
export function createSubAgentContext(prompt: string, parentMessage: any) {
    // fork logic from qv2
    const toolUse = parentMessage.message.content.find((c: any) =>
        c.type === "tool_use" && c.name === "AgentTool" && c.input?.prompt === prompt
    );

    if (!toolUse) {
        logger.error(`Could not find matching AgentTool tool use for prompt: ${prompt.slice(0, 50)}...`);
        return [{ type: "user", message: { content: prompt } }];
    }

    const contextHeader = `### FORKING CONVERSATION CONTEXT ###
### ENTERING SUB-AGENT ROUTINE ###
Entered sub-agent context

PLEASE NOTE: 
- The messages above this point are from the main thread prior to sub-agent execution. They are provided as context only.
- Context messages may include tool_use blocks for tools that are not available in the sub-agent context. You should only use the tools specifically provided to you in the system prompt.
- Only complete the specific sub-agent task you have been assigned below.`;

    const forkedParent = {
        ...parentMessage,
        message: {
            ...parentMessage.message,
            content: [toolUse]
        }
    };

    const toolResult = {
        type: "tool_result",
        tool_use_id: toolUse.id,
        content: [{ type: "text", text: contextHeader }],
        toolUseResult: {
            status: "sub_agent_entered",
            description: "Entered sub-agent context",
            message: contextHeader
        }
    };

    const userPrompt = {
        type: "user",
        message: { content: prompt }
    };

    return [forkedParent, toolResult, userPrompt];
}
