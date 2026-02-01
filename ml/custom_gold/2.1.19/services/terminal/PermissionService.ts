/**
 * File: src/services/terminal/PermissionService.ts
 * Role: UTILITY_HELPER
 * Service for handling tool permissions and policy enforcement.
 */

import { terminalLog } from "../../utils/shared/runtime.js";
import { getToolSettings, updateSettingsForSource } from "../../services/config/SettingsService.js";
import { hookService } from "../hooks/HookService.js";

export interface PermissionResponse {
    behavior: "passthrough" | "allow" | "deny" | "ask";
    suggestions?: any[];
    blockedPath?: string | null;
    message?: string;
    updatedInput?: any;
    decisionReason?: PermissionContext;
}

export interface PermissionContext {
    type: "rule" | "mode" | "subcommandResults" | "permissionPromptTool" | "hook" | "asyncAgent" | "sandboxOverride" | "classifier" | "workingDir" | "other" | "user_interaction";
    reason?: string;
    hookName?: string;
}

export interface ToolPermissionContext {
    mode: "prompt" | "bypassPermissions" | "plan";
    alwaysAllowRules?: {
        command?: string[]; // Allowed commands
        path?: string[];    // Allowed paths
    };
    additionalWorkingDirectories?: Map<string, string>;
    [key: string]: any;
}

// In-memory store for session-scoped permissions
const sessionPermissions = new Set<string>();

/**
 * Generates a unique rule key for a tool and its input.
 */
function getRuleKey(toolName: string, input: any): string {
    if (toolName === "Bash" || toolName === "mcp:Bash") {
        const cmd = input?.command?.trim();
        return cmd ? `Bash(${cmd})` : "";
    }
    const path = input?.file_path || input?.path;
    return path ? `${toolName}(${path})` : "";
}

/**
 * Checks tool permissions against policies and user decisions.
 * Corresponds to `$_` in chunks.
 */
export async function checkToolPermissions(
    toolName: string,
    input: any,
    context: { toolPermissionContext: ToolPermissionContext, [key: string]: any },
    assistantMessage?: any,
    toolUseId?: string
): Promise<PermissionResponse> {
    const permContext = context.toolPermissionContext;

    // 0. Hook: PreToolUse
    const hookResults = await hookService.dispatch("PreToolUse", {
        hook_event_name: "PreToolUse",
        tool_name: toolName,
        tool_input: input,
        tool_use_id: toolUseId
    });

    // Check for general blocks
    const blocked = hookResults.find(r => r.decision === 'block');
    if (blocked) {
        return {
            behavior: "deny",
            message: blocked.reason || "Blocked by hook",
            decisionReason: { type: "hook", reason: blocked.reason }
        };
    }

    // Check for specific permission decisions
    for (const res of hookResults) {
        if (!res.hookSpecificOutput || res.hookSpecificOutput.hookEventName !== "PreToolUse") continue;
        const out = res.hookSpecificOutput;

        if (out.permissionDecision === "deny") {
            return {
                behavior: "deny",
                message: out.permissionDecisionReason || "Denied by PreToolUse hook",
                decisionReason: { type: "hook", reason: out.permissionDecisionReason }
            };
        }
        if (out.permissionDecision === "allow") {
            return {
                behavior: "allow",
                updatedInput: out.updatedInput,
                decisionReason: { type: "hook", reason: out.permissionDecisionReason }
            };
        }
        if (out.permissionDecision === "ask") {
            return {
                behavior: "ask",
                updatedInput: out.updatedInput,
                decisionReason: { type: "hook", reason: out.permissionDecisionReason }
            };
        }
    }

    // 1. Check Global Permission Mode
    if (permContext.mode === "bypassPermissions") {
        return {
            behavior: "allow",
            decisionReason: { type: "mode", reason: "Permissions bypassed by mode settings." }
        };
    }

    const ruleKey = getRuleKey(toolName, input);
    if (!ruleKey) {
        return { behavior: "ask", suggestions: [], blockedPath: null };
    }

    // 2. Check "Always Allow" Rules (Persistent)
    const settings = getToolSettings("userSettings");
    const allowedRules = settings.permissions?.allow || [];
    const deniedRules = settings.permissions?.deny || [];

    if (deniedRules.includes(ruleKey)) {
        return {
            behavior: "deny",
            message: `Execution blocked by persistent deny rule: ${ruleKey}`,
            decisionReason: { type: "rule", reason: "Blocked by persistent rule." }
        };
    }

    if (allowedRules.includes(ruleKey)) {
        return {
            behavior: "allow",
            decisionReason: { type: "rule", reason: "Allowed by persistent rule." }
        };
    }

    // 3. Check Session Rules (In-memory)
    if (sessionPermissions.has(ruleKey)) {
        return {
            behavior: "allow",
            decisionReason: { type: "rule", reason: "Allowed for this session." }
        };
    }

    // 4. Check Context-specific Rules
    if (toolName === "Bash" || toolName === "mcp:Bash") {
        const cmd = input?.command?.trim();
        if (cmd && permContext.alwaysAllowRules?.command?.includes(cmd)) {
            return {
                behavior: "allow",
                decisionReason: { type: "rule", reason: "Allowed by context-specific command rule." }
            };
        }
    } else {
        const path = input?.file_path || input?.path;
        if (path && permContext.alwaysAllowRules?.path?.includes(path)) {
            return {
                behavior: "allow",
                decisionReason: { type: "rule", reason: "Allowed by context-specific path rule." }
            };
        }
    }

    // 5. Default: Ask User
    return {
        behavior: "ask",
        suggestions: [],
        blockedPath: null
    };
}

/**
 * Formats the decision reason.
 */
export function formatDecisionReason(decision: PermissionContext | null | undefined): string | undefined {
    if (!decision) return undefined;
    switch (decision.type) {
        case "rule": return "Allowed by rule";
        case "mode": return "Allowed by mode";
        case "subcommandResults": return "Allowed based on subcommand results";
        case "permissionPromptTool": return "Allowed by permission tool";
        case "user_interaction": return decision.reason || "User allowed";
        default: return decision.reason || "Automatic decision";
    }
}

/**
 * Handles the permission response from the UI, updates persistence if needed, and returns the formatted response.
 */
export function handlePermissionResponse(
    response: any,
    toolName: string,
    input: any,
    context: any
): PermissionResponse {
    const behavior = response.behavior || "ask";
    const scope = response.scope; // 'once', 'session', or 'always'
    const ruleKey = getRuleKey(toolName, input);

    if (behavior === "allow" && ruleKey) {
        if (scope === "session") {
            sessionPermissions.add(ruleKey);
        } else if (scope === "always") {
            const settings = getToolSettings("userSettings");
            const currentAllow = settings.permissions?.allow || [];
            if (!currentAllow.includes(ruleKey)) {
                updateSettingsForSource("userSettings", (current) => ({
                    ...current,
                    permissions: {
                        ...current.permissions,
                        allow: [...currentAllow, ruleKey]
                    }
                }));
            }
        }
    } else if (behavior === "deny" && ruleKey && scope === "always") {
        const settings = getToolSettings("userSettings");
        const currentDeny = settings.permissions?.deny || [];
        if (!currentDeny.includes(ruleKey)) {
            updateSettingsForSource("userSettings", (current) => ({
                ...current,
                permissions: {
                    ...current.permissions,
                    deny: [...currentDeny, ruleKey]
                }
            }));
        }
    }

    return {
        behavior: behavior,
        message: response.message,
        decisionReason: {
            type: "user_interaction",
            reason: response.message || (behavior === "allow" ? "User allowed" : "User denied")
        }
    };
}

// Aliases for deobfuscation
export const $_ = checkToolPermissions;
export const Rn2 = formatDecisionReason;
export const kdA = handlePermissionResponse;
