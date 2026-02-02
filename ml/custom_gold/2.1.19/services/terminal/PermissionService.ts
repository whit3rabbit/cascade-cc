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
    const normalizedName = toolName.startsWith("mcp:") ? toolName.slice(4) : toolName;

    switch (normalizedName) {
        case "Bash": {
            const cmd = input?.command?.trim();
            return cmd ? `bash:${cmd}` : "bash:*";
        }
        case "edit_file":
        case "repl_replace":
        case "replace_file_content":
        case "multi_replace_file_content":
        case "write_to_file": {
            const path = input?.file_path || input?.path || input?.TargetFile;
            return path ? `write:${path}` : "write:*";
        }
        case "read_file":
        case "view_file":
        case "list_dir":
        case "find_by_name":
        case "grep_search": {
            const path = input?.file_path || input?.path || input?.AbsolutePath || input?.SearchDirectory || input?.SearchPath;
            return path ? `read:${path}` : "read:*";
        }
        default:
            return `${normalizedName}:${JSON.stringify(input)}`;
    }
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

    // 0. Pre-check: bypass mode
    if (permContext.mode === "bypassPermissions") {
        return {
            behavior: "allow",
            decisionReason: { type: "mode", reason: "Permissions bypassed by mode settings." }
        };
    }

    // 1. Hook: PreToolUse
    const hookResults = await hookService.dispatch("PreToolUse", {
        hook_event_name: "PreToolUse",
        tool_name: toolName,
        tool_input: input,
        tool_use_id: toolUseId
    });

    for (const res of hookResults) {
        if (!res.hookSpecificOutput || res.hookSpecificOutput.hookEventName !== "PreToolUse") continue;
        const out = res.hookSpecificOutput;
        if (out.permissionDecision) {
            return {
                behavior: out.permissionDecision,
                message: out.permissionDecisionReason,
                updatedInput: out.updatedInput,
                decisionReason: { type: "hook", reason: out.permissionDecisionReason }
            };
        }
    }

    const ruleKey = getRuleKey(toolName, input);

    // 2. Check "Always Allow" Rules in Context (e.g. for allowed directories)
    if (toolName.includes("Bash")) {
        const cmd = input?.command?.trim();
        if (cmd && permContext.alwaysAllowRules?.command?.includes(cmd)) {
            return { behavior: "allow", decisionReason: { type: "rule", reason: "Context allow rule." } };
        }
    } else {
        const path = input?.file_path || input?.path || input?.TargetFile || input?.AbsolutePath;
        if (path && permContext.alwaysAllowRules?.path?.some((p: string) => path.startsWith(p))) {
            return { behavior: "allow", decisionReason: { type: "rule", reason: "Context path allow rule." } };
        }
    }

    // 3. Check Persistent Rules (SettingsService)
    const settings = getToolSettings("userSettings");
    const allowRules = settings.permissions?.allow || [];
    const denyRules = settings.permissions?.deny || [];

    if (denyRules.includes(ruleKey)) {
        return { behavior: "deny", message: `Blocked by rule: ${ruleKey}`, decisionReason: { type: "rule" } };
    }
    if (allowRules.includes(ruleKey)) {
        return { behavior: "allow", decisionReason: { type: "rule" } };
    }

    // 4. Check Session Rules
    if (sessionPermissions.has(ruleKey)) {
        return { behavior: "allow", decisionReason: { type: "rule", reason: "Session allowed." } };
    }

    // 5. Default: Ask
    return { behavior: "ask", suggestions: [], blockedPath: null };
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

/**
 * Updates the tool permission context with new rules or mode changes.
 * Corresponds to `WY` in chunks.
 */
export function updateToolPermissionContext(
    current: ToolPermissionContext,
    update: any
): ToolPermissionContext {
    const next = { ...current };

    if (update.type === "addRules") {
        const type = update.behavior === "allow" ? "allow" : "deny";
        // Context rules (ephemeral for current task/session)
        if (!next.alwaysAllowRules) next.alwaysAllowRules = {};
        const list = update.rules || [];
        if (update.isPath) {
            next.alwaysAllowRules.path = [...(next.alwaysAllowRules.path || []), ...list];
        } else {
            next.alwaysAllowRules.command = [...(next.alwaysAllowRules.command || []), ...list];
        }
    } else if (update.type === "setMode") {
        next.mode = update.mode;
    }

    return next;
}

// Aliases for deobfuscation
export const $_ = checkToolPermissions;
export const Rn2 = formatDecisionReason;
export const kdA = handlePermissionResponse;
export const WY = updateToolPermissionContext;
