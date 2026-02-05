/**
 * File: src/services/terminal/PermissionService.ts
 * Role: UTILITY_HELPER
 * Service for handling tool permissions and policy enforcement.
 */

import { terminalLog } from "../../utils/shared/runtime.js";
import { getSettings, getToolSettings, updateSettingsForSource } from "../../services/config/SettingsService.js";
import { hookService } from "../hooks/HookService.js";
import { isBubblewrapSandbox, isDocker } from "../../utils/shared/runtimeAndEnv.js";
import micromatch from 'micromatch';
import path from 'path';

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
    mode: "default" | "acceptEdits" | "plan" | "dontAsk" | "bypassPermissions" | "delegate";
    alwaysAllowRules?: {
        command?: string[]; // Allowed commands
        path?: string[];    // Allowed paths
    };
    additionalWorkingDirectories?: Map<string, string>;
    projectSettingsPath?: string;
    [key: string]: any;
}

// In-memory store for session-scoped permissions
const sessionPermissions = new Set<string>();

const TOOL_TYPE_MAPPING: Record<string, string> = {
    'run_command': 'Bash',
    'read_url_content': 'WebFetch',
    'search_web': 'WebFetch',
    'read_resource': 'Read', // MCP read
    // Read tools
    'read_file': 'Read',
    'view_file': 'Read',
    'list_dir': 'Read',
    'find_by_name': 'Read',
    'grep_search': 'Read',
    'view_file_outline': 'Read',
    'view_code_item': 'Read',
    'view_content_chunk': 'Read',
    // Edit tools
    'edit_file': 'Edit',
    'replace_file_content': 'Edit',
    'multi_replace_file_content': 'Edit',
    'write_to_file': 'Edit',
};

function getRuleKey(toolName: string, input: any): string {
    const ruleType = TOOL_TYPE_MAPPING[toolName] || toolName;
    if (ruleType === 'Bash') return `Bash(${input?.CommandLine || input?.command || ''})`;
    if (ruleType === 'Read' || ruleType === 'Edit') {
        const p = input?.AbsolutePath || input?.TargetFile || input?.SearchDirectory || input?.SearchPath || input?.StartDirectory || input?.path || input?.file_path;
        return `${ruleType}(${p})`;
    }
    return `${toolName}:${JSON.stringify(input)}`;
}

function matchBashRule(rule: string, command: string): boolean {
    const match = rule.match(/^Bash(?:\((.*)\))?$/);
    if (!match) return false;

    const pattern = match[1];
    if (pattern === undefined) return true;
    if (pattern === '*') return true;

    return (micromatch as any).isMatch(command, pattern);
}

function matchFileRule(rule: string, filePath: string, toolType: 'Read' | 'Edit', settingsPath?: string): boolean {
    const regex = new RegExp(`^${toolType}(?:\\((.*)\\))?$`);
    const match = rule.match(regex);
    if (!match) return false;

    const patternContent = match[1];
    if (patternContent === undefined || patternContent === '*') return true;

    // //path -> absolute
    if (patternContent.startsWith('//')) {
        const absPattern = patternContent.slice(1);
        return (micromatch as any).isMatch(filePath, absPattern);
    }

    // ~/path -> home
    if (patternContent.startsWith('~/')) {
        const home = process.env.HOME || '';
        const absPattern = path.join(home, patternContent.slice(2));
        return (micromatch as any).isMatch(filePath, absPattern);
    }

    // /path -> relative to settings file
    if (patternContent.startsWith('/') && settingsPath) {
        const settingsDir = path.dirname(settingsPath);
        const absPattern = path.join(settingsDir, patternContent.slice(1));
        return (micromatch as any).isMatch(filePath, absPattern);
    }

    // path -> relative to cwd or simple glob
    const cwd = process.cwd();
    const relPath = path.relative(cwd, filePath);
    const cleanPattern = patternContent.startsWith('./') ? patternContent.slice(2) : patternContent;

    return (micromatch as any).isMatch(relPath, cleanPattern);
}

function matchWebFetchRule(rule: string, url: string): boolean {
    const match = rule.match(/^WebFetch(?:\((.*)\))?$/);
    if (!match) return false;

    const pattern = match[1];
    if (pattern === undefined) return true;

    if (pattern.startsWith('domain:')) {
        const allowedDomain = pattern.slice(7);
        try {
            const parsed = new URL(url);
            return parsed.hostname === allowedDomain || parsed.hostname.endsWith('.' + allowedDomain);
        } catch (e) {
            return false;
        }
    }

    return false;
}

function matchMcpRule(rule: string, toolName: string): boolean {
    if (!toolName.startsWith('mcp__')) return false;

    if (rule.endsWith('__*')) {
        const prefix = rule.slice(0, -3);
        return toolName.startsWith(prefix);
    }

    if (rule.split('__').length === 2 && !rule.includes('*')) {
        return toolName.startsWith(rule + '__');
    }

    return toolName === rule;
}

function doesRuleMatch(rule: string, toolName: string, input: any, context: ToolPermissionContext): boolean {
    const normalizedTool = TOOL_TYPE_MAPPING[toolName] || toolName;
    const isMcp = toolName.startsWith('mcp__');

    if (normalizedTool === 'Bash') {
        if (!rule.startsWith('Bash')) return false;
        const cmd = input?.CommandLine || input?.command || '';
        return matchBashRule(rule, cmd);
    }

    if (normalizedTool === 'Read' || normalizedTool === 'Edit') {
        if (!rule.startsWith(normalizedTool)) return false;
        const p = input?.AbsolutePath || input?.TargetFile || input?.SearchDirectory || input?.SearchPath || input?.StartDirectory || input?.path || input?.file_path;
        if (!p) return false;
        return matchFileRule(rule, p, normalizedTool, context.projectSettingsPath);
    }

    if (normalizedTool === 'WebFetch') {
        if (!rule.startsWith('WebFetch')) return false;
        const url = input?.Url || input?.url || '';
        if (!url) return false;
        return matchWebFetchRule(rule, url);
    }

    if (isMcp) {
        return matchMcpRule(rule, toolName);
    }

    if (rule === toolName) return true;

    const match = rule.match(/^([^(]+)(?:\((.*)\))?$/);
    if (!match) return false;
    const ruleTool = match[1];
    const ruleSpec = match[2];

    if (ruleTool !== normalizedTool) return false;
    if (!ruleSpec) return true;

    return false;
}

function checkContextAllowRules(toolName: string, input: any, context: ToolPermissionContext): boolean {
    if (!context.alwaysAllowRules) return false;

    const normalizedTool = TOOL_TYPE_MAPPING[toolName] || toolName;

    if (normalizedTool === 'Bash' && context.alwaysAllowRules.command) {
        const cmd = input?.CommandLine || input?.command || '';
        if (context.alwaysAllowRules.command.includes(cmd)) return true;
    }

    if ((normalizedTool === 'Read' || normalizedTool === 'Edit') && context.alwaysAllowRules.path) {
        const p = input?.AbsolutePath || input?.TargetFile || input?.SearchDirectory || input?.SearchPath || input?.StartDirectory || input?.path || input?.file_path;
        if (!p) return false;
        for (const allowedPath of context.alwaysAllowRules.path) {
            if (p.startsWith(allowedPath)) return true;
        }
    }

    return false;
}

export async function checkToolPermissions(
    toolName: string,
    input: any,
    context: { toolPermissionContext: ToolPermissionContext, [key: string]: any },
    assistantMessage?: any,
    toolUseId?: string
): Promise<PermissionResponse> {
    const permContext = context.toolPermissionContext;

    // 0. Mode: Bypass
    if (permContext.mode === "bypassPermissions") {
        return {
            behavior: "allow",
            decisionReason: { type: "mode", reason: "Permissions bypassed by mode settings." }
        };
    }

    // 0.1 Mode: Delegate
    if (permContext.mode === "delegate") {
        const allowedDelegateTools = ["TeammateTool", "TaskCreate", "TaskGet", "TaskUpdate", "TaskList"];
        const normalizedName = toolName.startsWith("mcp:") ? toolName.slice(4) : toolName;
        if (allowedDelegateTools.includes(normalizedName)) {
            return { behavior: "allow", decisionReason: { type: "mode", reason: "Allowed in delegate mode." } };
        }
        return { behavior: "deny", message: `Tool ${toolName} is not allowed in delegate mode.`, decisionReason: { type: "mode", reason: "Blocked by delegate mode." } };
    }

    // 0.2 Mode: Plan
    if (permContext.mode === "plan") {
        const normalizedTool = TOOL_TYPE_MAPPING[toolName] || toolName;
        if (normalizedTool === 'Edit' || normalizedTool === 'Bash') {
            return { behavior: "deny", message: "Modification tools are disabled in Plan mode.", decisionReason: { type: "mode", reason: "Plan mode restriction." } };
        }
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

    // 1.5 Context Allow Rules (Ephemeral)
    if (checkContextAllowRules(toolName, input, permContext)) {
        return { behavior: "allow", decisionReason: { type: "rule", reason: "Context allow rule." } };
    }

    // 2. Persistent/Managed Rules
    const settings = getSettings();
    const permissions = settings.permissions || getToolSettings("userSettings").permissions || {};

    const denyRules = permissions.deny || [];
    const askRules = permissions.ask || [];
    const allowRules = permissions.allow || [];

    // Deny
    for (const rule of denyRules) {
        if (doesRuleMatch(rule, toolName, input, permContext)) {
            return { behavior: "deny", message: `Blocked by rule: ${rule}`, decisionReason: { type: "rule" } };
        }
    }

    // Ask
    for (const rule of askRules) {
        if (doesRuleMatch(rule, toolName, input, permContext)) {
            return { behavior: "ask", blockedPath: null, decisionReason: { type: "rule", reason: "Explicit Ask rule." } };
        }
    }

    // Allow
    for (const rule of allowRules) {
        if (doesRuleMatch(rule, toolName, input, permContext)) {
            return { behavior: "allow", decisionReason: { type: "rule" } };
        }
    }

    // 3. Session Rules (Explicit "Don't ask again")
    const implicitRuleKey = getRuleKey(toolName, input);
    if (sessionPermissions.has(implicitRuleKey)) {
        return { behavior: "allow", decisionReason: { type: "rule", reason: "Session allowed." } };
    }

    // 4. Sandbox Override
    if (isBubblewrapSandbox() || isDocker()) {
        const normalizedTool = TOOL_TYPE_MAPPING[toolName] || toolName;
        if (normalizedTool === 'Bash') {
            return { behavior: "allow", decisionReason: { type: "sandboxOverride", reason: "Container/Sandbox detected." } };
        }
    }

    // 5. Default Behavior
    if (permContext.mode === 'dontAsk') {
        return { behavior: "deny", message: "Blocked by dontAsk mode.", decisionReason: { type: "mode" } };
    }

    const normalizedTool = TOOL_TYPE_MAPPING[toolName] || toolName;

    if (normalizedTool === 'Read') {
        return { behavior: "allow", decisionReason: { type: "classifier", reason: "Read-only tool default allow." } };
    }

    if (normalizedTool === 'Edit') {
        if (permContext.mode === 'acceptEdits') {
            return { behavior: "allow", decisionReason: { type: "mode", reason: "acceptEdits mode." } };
        }
        return { behavior: "ask", blockedPath: null };
    }

    if (normalizedTool === 'Bash') {
        return { behavior: "ask", blockedPath: null };
    }

    return { behavior: "ask", blockedPath: null };
}

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

export function handlePermissionResponse(
    response: any,
    toolName: string,
    input: any,
    context: any
): PermissionResponse {
    const behavior = response.behavior || "ask";
    const scope = response.scope;

    const ruleKey = getRuleKey(toolName, input);

    if (behavior === "allow" && ruleKey) {
        if (scope === "session") {
            sessionPermissions.add(ruleKey);
        } else if (scope === "always") {
            updateSettingsForSource("userSettings", (current) => ({
                ...current,
                permissions: {
                    ...current.permissions,
                    allow: [...(current.permissions?.allow || []), ruleKey]
                }
            }));
        }
    } else if (behavior === "deny" && ruleKey && scope === "always") {
        updateSettingsForSource("userSettings", (current) => ({
            ...current,
            permissions: {
                ...current.permissions,
                deny: [...(current.permissions?.deny || []), ruleKey]
            }
        }));
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

export function updateToolPermissionContext(
    current: ToolPermissionContext,
    update: any
): ToolPermissionContext {
    const next = { ...current };

    if (update.type === "addRules") {
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

export const $_ = checkToolPermissions;
export const Rn2 = formatDecisionReason;
export const kdA = handlePermissionResponse;
export const WY = updateToolPermissionContext;
