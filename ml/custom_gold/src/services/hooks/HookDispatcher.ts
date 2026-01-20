
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { log } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { createAttachment } from "../terminal/MessageFactory.js";
import { getGlobalState } from "../session/globalState.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { runParallel } from "../../utils/shared/runParallel.js";
import { AsyncQueue } from "../../utils/shared/AsyncQueue.js";

const logger = log("hooks");

// Schema for hook output (wJ7/HD1)
const HookOutputSchema = z.object({
    continue: z.boolean().optional(),
    suppressOutput: z.boolean().optional(),
    stopReason: z.string().optional(),
    decision: z.enum(["approve", "block"]).optional(),
    reason: z.string().optional(),
    systemMessage: z.string().optional(),
    hookSpecificOutput: z.discriminatedUnion("hookEventName", [
        z.object({
            hookEventName: z.literal("PreToolUse"),
            permissionDecision: z.enum(["allow", "deny", "ask"]).optional(),
            permissionDecisionReason: z.string().optional(),
            updatedInput: z.record(z.string(), z.unknown()).optional()
        }),
        z.object({
            hookEventName: z.literal("UserPromptSubmit"),
            additionalContext: z.string().optional()
        }),
        z.object({
            hookEventName: z.literal("SessionStart"),
            additionalContext: z.string().optional()
        }),
        z.object({
            hookEventName: z.literal("SubagentStart"),
            additionalContext: z.string().optional()
        }),
        z.object({
            hookEventName: z.literal("PostToolUse"),
            additionalContext: z.string().optional(),
            updatedMCPToolOutput: z.unknown().optional()
        }),
        z.object({
            hookEventName: z.literal("PostToolUseFailure"),
            additionalContext: z.string().optional()
        }),
        z.object({
            hookEventName: z.literal("PermissionRequest"),
            decision: z.union([
                z.object({
                    behavior: z.literal("allow"),
                    updatedInput: z.record(z.string(), z.unknown()).optional()
                }),
                z.object({
                    behavior: z.literal("deny"),
                    message: z.string().optional(),
                    interrupt: z.boolean().optional()
                })
            ])
        })
    ]).optional()
});

export type HookOutput = z.infer<typeof HookOutputSchema>;

/**
 * Combines multiple abort signals into one. (tb)
 */
export function combineAbortSignals(signal1: AbortSignal, signal2?: AbortSignal) {
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    signal1.addEventListener("abort", onAbort);
    signal2?.addEventListener("abort", onAbort);
    return {
        signal: controller.signal,
        cleanup: () => {
            signal1.removeEventListener("abort", onAbort);
            signal2?.removeEventListener("abort", onAbort);
        }
    };
}

/**
 * Validates and parses hook stdout. (ZK9)
 */
export function validateHookOutput(stdout: string) {
    const trimmed = stdout.trim();
    if (!trimmed.startsWith("{")) {
        return { plainText: stdout };
    }
    try {
        const parsed = JSON.parse(trimmed);
        const result = HookOutputSchema.safeParse(parsed);
        if (result.success) {
            return { json: result.data };
        } else {
            const errorMsg = `Hook JSON validation failed: ${result.error.issues.map((e: any) => `${e.path.join(".")}: ${e.message}`).join(", ")}`;
            logger.warn(errorMsg);
            return { plainText: stdout, validationError: errorMsg };
        }
    } catch (e) {
        return { plainText: stdout };
    }
}

/**
 * Result of a hook execution.
 */
interface HookResult {
    message?: any;
    preventContinuation?: boolean;
    stopReason?: string;
    permissionBehavior?: string;
    blockingError?: { blockingError: string; command: string };
    systemMessage?: string;
    additionalContext?: string;
    updatedMCPToolOutput?: any;
    outcome?: "success" | "cancelled" | "non_blocking_error";
    hook?: any;
    stdout?: string;
    stderr?: string;
    exitCode?: number;
}

/**
 * Executes a single hook command. (FD1)
 */
export async function executeHookCommand(
    hook: any,
    hookEvent: string,
    hookName: string,
    hookInput: string,
    signal: AbortSignal,
    index: number
) {
    const projectDir = getProjectRoot();
    const env = {
        ...process.env,
        CLAUDE_PROJECT_DIR: projectDir
    };

    const timeoutMs = (hook.timeout || 60) * 1000;
    const { signal: combinedSignal, cleanup } = combineAbortSignals(AbortSignal.timeout(timeoutMs), signal);

    try {
        const child = spawn(hook.command, [], {
            env,
            shell: true,
            stdio: ["pipe", "pipe", "pipe"]
        });

        let stdout = "";
        let stderr = "";

        child.stdout.on("data", (data: any) => stdout += data.toString());
        child.stderr.on("data", (data: any) => stderr += data.toString());

        return new Promise<any>((resolve, reject) => {
            child.on("close", (code: any) => {
                cleanup();
                resolve({
                    stdout,
                    stderr,
                    status: code ?? 1,
                    aborted: signal.aborted
                });
            });
            child.on("error", (err: any) => {
                cleanup();
                reject(err);
            });
            child.stdin.write(hookInput);
            child.stdin.end();

            combinedSignal.addEventListener("abort", () => {
                child.kill();
                cleanup();
                resolve({
                    stdout,
                    stderr,
                    status: 1,
                    aborted: true
                });
            });
        });
    } catch (err) {
        cleanup();
        throw err;
    }
}

/**
 * Processes the result of a hook execution. (YK9)
 */
export function processHookResult(params: {
    json?: HookOutput,
    command: string,
    hookName: string,
    toolUseID: string,
    hookEvent: string,
    stdout: string,
    stderr: string,
    exitCode: number
}): HookResult {
    const { json, hookName, toolUseID, hookEvent, stdout, stderr, exitCode } = params;
    const result: any = {};

    if (json) {
        if (json.continue === false) {
            result.preventContinuation = true;
            result.stopReason = json.stopReason;
        }
        if (json.decision) {
            result.permissionBehavior = json.decision === "approve" ? "allow" : "deny";
            if (json.decision === "block") {
                result.blockingError = { blockingError: json.reason || "Blocked by hook", command: params.command };
            }
        }
        if (json.systemMessage) result.systemMessage = json.systemMessage;

        if (json.hookSpecificOutput) {
            const hso = json.hookSpecificOutput;
            switch (hso.hookEventName) {
                case "PreToolUse":
                    if (hso.permissionDecision) {
                        result.permissionBehavior = hso.permissionDecision;
                        if (hso.permissionDecision === "deny") {
                            result.blockingError = {
                                blockingError: hso.permissionDecisionReason || json.reason || "Blocked by hook",
                                command: params.command
                            };
                        }
                    }
                    result.hookPermissionDecisionReason = hso.permissionDecisionReason || json.reason;
                    if (hso.updatedInput) result.updatedInput = hso.updatedInput;
                    break;
                case "PostToolUse":
                    result.additionalContext = hso.additionalContext;
                    if (hso.updatedMCPToolOutput) result.updatedMCPToolOutput = hso.updatedMCPToolOutput;
                    break;
                // ... handle other hook types if needed
                default:
                    if ("additionalContext" in hso) result.additionalContext = hso.additionalContext;
                    break;
            }
        }
    }

    return {
        ...result,
        message: result.blockingError ? createAttachment({
            type: "hook_blocking_error",
            hookName,
            toolUseID,
            hookEvent,
            blockingError: result.blockingError
        }) : createAttachment({
            type: "hook_success",
            hookName,
            toolUseID,
            hookEvent,
            content: "Success",
            stdout,
            stderr,
            exitCode
        })
    };
}

/**
 * Orchestrates the execution of hook commands. (ms)
 */
export async function* runHookCommands(params: {
    hookInput: any,
    toolUseID: string,
    matchQuery?: string,
    signal?: AbortSignal,
    timeoutMs?: number,
    toolUseContext: any,
    messages?: any[]
}) {
    const { hookInput, toolUseID, matchQuery, signal, toolUseContext, messages } = params;
    const hookEvent = hookInput.hook_event_name;
    const hookName = matchQuery ? `${hookEvent}:${matchQuery}` : hookEvent;

    // 1. Get matching hooks (stub for now, needs logic from OJ7/qM0)
    const matchedHooks: any[] = [];

    if (matchedHooks.length === 0) return;

    logTelemetryEvent("tengu_run_hook", {
        hookName,
        numCommands: matchedHooks.length
    });

    // 2. Report progress
    for (const hook of matchedHooks) {
        yield {
            message: {
                type: "progress",
                data: {
                    type: "hook_progress",
                    hookEvent,
                    hookName,
                    command: hook.command || hook.prompt || "callback",
                    statusMessage: hook.statusMessage
                },
                toolUseID,
                parentToolUseID: toolUseID,
                timestamp: new Date().toISOString(),
                uuid: randomUUID()
            }
        };
    }

    // 3. Execute hooks (parallel execution using runParallel/xHA)
    const hookInputsStr = JSON.stringify(hookInput);
    const executionTasks: AsyncGenerator<HookResult>[] = matchedHooks.map(async function* (hook, index) {
        if (hook.type === "callback") {
            // handle callback hook ...
            return;
        }

        if (hook.type === "command") {
            const result = await executeHookCommand(hook, hookEvent, hookName, hookInputsStr, signal || new AbortController().signal, index);
            if (result.aborted) {
                yield {
                    message: createAttachment({
                        type: "hook_cancelled",
                        hookName,
                        toolUseID,
                        hookEvent
                    }),
                    outcome: "cancelled",
                    hook
                } as HookResult;
                return;
            }

            const { json, validationError } = validateHookOutput(result.stdout);
            if (validationError) {
                yield {
                    message: createAttachment({
                        type: "hook_non_blocking_error",
                        hookName,
                        toolUseID,
                        hookEvent,
                        stderr: validationError,
                        stdout: result.stdout,
                        exitCode: 1
                    }),
                    outcome: "non_blocking_error",
                    hook
                } as HookResult;
                return;
            }

            const processed = processHookResult({
                json,
                command: hook.command,
                hookName,
                toolUseID,
                hookEvent,
                stdout: result.stdout,
                stderr: result.stderr,
                exitCode: result.status
            });

            yield { ...processed, outcome: "success", hook } as HookResult;
        }
    });

    for await (const result of runParallel<HookResult>(executionTasks, 10)) {
        if (result.preventContinuation) yield { preventContinuation: true, stopReason: result.stopReason };
        if (result.blockingError) yield { blockingError: result.blockingError };
        if (result.message) yield { message: result.message };
        if (result.systemMessage) {
            yield {
                message: createAttachment({
                    type: "hook_system_message",
                    content: result.systemMessage,
                    hookName,
                    toolUseID,
                    hookEvent
                })
            };
        }
        if (result.additionalContext) yield { additionalContexts: [result.additionalContext] };
        if (result.updatedMCPToolOutput) yield { updatedMCPToolOutput: result.updatedMCPToolOutput };
        // ... handle permission behavior
    }
}
