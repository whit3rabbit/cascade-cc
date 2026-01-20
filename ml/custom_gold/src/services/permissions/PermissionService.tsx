
import React, { useState, useCallback, useMemo, useEffect } from 'react';
// @ts-ignore
import { Box, Text, useInput } from 'ink';
import { useAppState } from '../../contexts/AppStateContext.js';
import { logEvent } from '../telemetry/TelemetryService.js';
import { getApiKeyWithSource, isApiKeyHelperTrusted } from '../auth/apiKeyManager.js';
import { isInteractiveAuthRequired, isApiKeyConfigured } from '../auth/authSource.js';
import { estimateTokens } from '../../utils/shared/tokenUtils.js';

// Logic from chunk_514.ts (Permissions & Remote Sessions)

interface PermissionRequestResult {
    behavior: "allow" | "deny" | "ask";
    message?: string;
    updatedInput?: any;
    userModified?: boolean;
    acceptFeedback?: string;
    decisionReason?: any;
}

interface ToolPermissionRequest {
    assistantMessage: any;
    tool: any; // Tool type
    description: string;
    input: any;
    toolUseContext: any;
    toolUseID: string;
    permissionResult: PermissionRequestResult;
    permissionPromptStartTimeMs: number;
    onAbort: () => void;
    onAllow: (newInput: any, updatedPermissions: any[], feedback?: string) => Promise<void>;
    onReject: (feedback?: string) => void;
    recheckPermission: () => Promise<void>;
}

// --- Tool Permission Request Hook (A07) ---
// Returns a callback 'requestToolPermission' that handles the flow
// Also maintains 'pendingRequests' state which can be rendered by the UI
export function useToolPermissionRequest(setPendingRequests: React.Dispatch<React.SetStateAction<ToolPermissionRequest[]>>, setPermissionContext: (ctx: any) => void) {
    return useCallback(async (tool: any, input: any, context: any, assistantMessage: any, toolUseID: string, options?: any) => {
        return new Promise<PermissionRequestResult>((resolve) => {
            function cancel() {
                logEvent("tengu_tool_use_cancelled", {
                    messageID: assistantMessage.message.id,
                    toolName: tool.name
                });
            }

            function finish(result: PermissionRequestResult, interrupt: boolean = false) {
                // Format message if needed
                resolve(result);
                if (!interrupt) context.abortController.abort(); // Actually logical flow from chunk_514
                // In chunk_514 logic: !H means if NOT interrupt/handled, abort.
                // Here verify logic carefully.
            }

            if (context.abortController.signal.aborted) {
                cancel();
                finish({ behavior: "ask", message: "Aborted" }); // simplistic
                return;
            }

            // This logic assumes we have a way to check permissions async first
            // usually checkPermission(tool, input, context...)
            // If check returns ALLOW, resolve immediately.
            // If ASK/DENY, proceed to UI prompt.

            // For now, stubbing the immediate check to simulate "ASK"
            const initialCheck: PermissionRequestResult = { behavior: "ask" };

            if (initialCheck.behavior === "allow") {
                // log approval
                resolve({ behavior: "allow", updatedInput: input, userModified: false });
                return;
            }

            // Handle DENY/ASK
            async function showPrompt() {
                const description = await tool.description(input, {
                    isNonInteractiveSession: context.options.isNonInteractiveSession,
                    toolPermissionContext: (await context.getAppState()).toolPermissionContext, // simplified
                    tools: context.options.tools
                });

                if (context.abortController.signal.aborted) {
                    cancel();
                    resolve({ behavior: "ask" });
                    return;
                }

                if (initialCheck.behavior === "deny") {
                    // Log denial
                    resolve(initialCheck);
                    return;
                }

                let handled = false;
                const startTime = Date.now();

                setPendingRequests(prev => [...prev, {
                    assistantMessage,
                    tool,
                    description,
                    input,
                    toolUseContext: context,
                    toolUseID,
                    permissionResult: initialCheck,
                    permissionPromptStartTimeMs: startTime,
                    onAbort() {
                        if (handled) return;
                        handled = true;
                        // Log user abort
                        cancel();
                        resolve({ behavior: "ask", message: "User aborted" });
                        // Abort controller
                        context.abortController.abort();
                    },
                    async onAllow(newInput, newPermissions, feedback) {
                        if (handled) return;
                        handled = true;
                        // permissions: { ...permissions, defaultMode: action.mode as any }(wLA) // This line is syntactically incorrect as code. Assuming it's a comment or placeholder.
                        // persist permissions logic (wLA)
                        // update state
                        // log granted
                        const userModified = tool.inputsEquivalent ? !tool.inputsEquivalent(input, newInput) : false;
                        resolve({
                            behavior: "allow",
                            updatedInput: newInput,
                            userModified,
                            acceptFeedback: feedback
                        });
                    },
                    onReject(feedback) {
                        if (handled) return;
                        handled = true;
                        // log rejection
                        resolve({ behavior: "deny", message: feedback }); // User rejected
                    },
                    async recheckPermission() {
                        if (handled) return;
                        // Re-run checkPermission logic
                        // If allow, resolve
                    }
                }]);
            }
            showPrompt();
        });
    }, [setPendingRequests, setPermissionContext]);
}


// --- Launch Remote Agent (lt2) ---
export async function launchRemoteAgent(prompt: string, prevMessages: any[], commands: any[], contextState: any) {
    logEvent("tengu_input_background", {});

    // 1. Check Git Status
    try {
        const cwd = process.cwd();
        const gitStatus = await getGitStatus(cwd);

        if (gitStatus.hasUncommitted) {
            console.log("Uncommitted changes detected. Remote agent requires a clean working directory or synced changes.");
            // In real impl, we'd prompt user here (Ww0)
            return {
                messages: [{ role: "assistant", content: "Cannot launch remote agent: Uncommitted changes detected. Please commit or stash your changes." }],
                shouldQuery: false
            };
        }

        if (gitStatus.hasUnpushed) {
            console.log("Unpushed commits detected. Remote agent needs to sync with remote.");
            // Prompt to push?
        }

    } catch (e) {
        console.error("Failed to check git status:", e);
    }

    // 2. Create session (Stub)
    console.log("Launching remote agent (Stub)...");

    return {
        messages: [{ role: "assistant", content: "Remote agent launching is currently a work in progress in this deobfuscated version." }],
        shouldQuery: false
    };
}

import { getGitStatus } from '../../utils/git/GitUtils.js';


// --- Token Counter (dt2) ---
export function useTokenCount(text: string) {
    return useMemo(() => {
        // cyA(A) logic
        const tokens = estimateTokens(text);
        let level = "low";
        if (tokens > 10000) level = "high";
        else if (tokens > 5000) level = "medium";

        return {
            level,
            tokens
        };
    }, [text]);
}



// --- API Key Status Hook (Tt2) ---
export function useApiKeyStatus() {
    const [status, setStatus] = useState<"valid" | "loading" | "missing" | "invalid" | "error">(() => {
        const { key } = getApiKeyWithSource();
        if (!isInteractiveAuthRequired() || isApiKeyConfigured()) return "valid";
        if (key) return "loading";
        return "missing";
    });

    const [error, setError] = useState<Error | null>(null);

    const reverify = useCallback(async () => {
        if (!isInteractiveAuthRequired() || isApiKeyConfigured()) {
            setStatus("valid");
            return;
        }
        const { key } = getApiKeyWithSource();
        if (!key) {
            setStatus("missing");
            return;
        }
        try {
            // Stub verification logic (St2)
            // const valid = await verifyApiKey(key);
            const valid = true;
            setStatus(valid ? "valid" : "invalid");
        } catch (err: any) {
            setError(err);
            setStatus("error");
        }
    }, []);

    return { status, reverify, error };
}

// --- Prompt Coaching Hook (Lt2) ---
export function usePromptCoaching(inputValue: string, isAssistantResponding: boolean) {
    // Stub logic
    useEffect(() => {
        // Logic to analyze input and suggest tips
    }, [inputValue, isAssistantResponding]);

    return {
        tip: null,
        dismissTip: () => { }
    };
}

// --- Todo Toggle (xt2) ---
export function useTodoToggle(todos: any[]) {
    const [state, setState] = useAppState();
    useInput((input, key) => {
        if (key.ctrl && input === "t") {
            logEvent("tengu_toggle_todos", {
                is_expanded: (state as any).showExpandedTodos,
                has_todos: todos && todos.length > 0
            });
            setState((prev: any) => ({ ...prev, showExpandedTodos: !(prev as any).showExpandedTodos }));
        }
    });
}

// --- Global Cancel (vt2) ---
export function useCancelRequest(setMessages: (fn: (msgs: any[]) => any[]) => void, clearPending: () => void, currentTool: any, mode: string, toolUseState: any, cancelCallback: () => void, inputMode: string, isProcessing: boolean) {
    const [state] = useAppState();
    // Queued commands check
    useInput((input, key) => {
        if (!key.escape) return;
        if (mode === "transcript") return;
        if (isProcessing) return;
        if (toolUseState?.aborted) return;
        // if (!currentTool) return; // Logic check
        // ...
        logEvent("tengu_cancel", {});
        setMessages(() => []); // clear
        clearPending();
    });
}
