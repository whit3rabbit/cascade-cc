import { z } from "zod";
import { SandboxConfigSchema } from "../sandbox/SandboxConfigSchema.js";
import { MarketplaceManifestSchema, PluginManifestSchema } from "../marketplace/MarketplaceSchemas.js";

// Helper schemas
const PermissionRuleSchema = z.string();
const PermissionModeEnum = z.enum(["prompt", "allow", "deny"]);
const MarketplaceSourceEnum = z.enum(["github", "url", "git", "directory", "file"]); // Placeholder for vTA

export const PermissionsSettingsSchema = z.object({
    allow: z.array(PermissionRuleSchema).optional().describe("List of permission rules for allowed operations"),
    deny: z.array(PermissionRuleSchema).optional().describe("List of permission rules for denied operations"),
    ask: z.array(PermissionRuleSchema).optional().describe("List of permission rules that should always prompt for confirmation"),
    defaultMode: PermissionModeEnum.optional().describe("Default permission mode when Claude Code needs access"),
    disableBypassPermissionsMode: z.enum(["disable"]).optional().describe("Disable the ability to bypass permission prompts"),
    additionalDirectories: z.array(z.string()).optional().describe("Additional directories to include in the permission scope")
}).passthrough();

export const CommandHookSchema = z.object({
    type: z.literal("command").describe("Bash command hook type"),
    command: z.string().describe("Shell command to execute"),
    timeout: z.number().positive().optional().describe("Timeout in seconds for this specific command"),
    statusMessage: z.string().optional().describe("Custom status message to display in spinner while hook runs")
});

export const PromptHookSchema = z.object({
    type: z.literal("prompt").describe("LLM prompt hook type"),
    prompt: z.string().describe("Prompt to evaluate with LLM. Use $ARGUMENTS placeholder for hook input JSON."),
    timeout: z.number().positive().optional().describe("Timeout in seconds for this specific prompt evaluation"),
    model: z.string().optional().describe('Model to use for this prompt hook (e.g., "claude-sonnet-4-5-20250929"). If not specified, uses the default small fast model.'),
    statusMessage: z.string().optional().describe("Custom status message to display in spinner while hook runs")
});

export const AgentHookSchema = z.object({
    type: z.literal("agent").describe("Agentic verifier hook type"),
    prompt: z.string().describe('Prompt describing what to verify (e.g. "Verify that unit tests ran and passed."). Use $ARGUMENTS placeholder for hook input JSON.'),
    timeout: z.number().positive().optional().describe("Timeout in seconds for agent execution (default 60)"),
    model: z.string().optional().describe('Model to use for this agent hook (e.g., "claude-sonnet-4-5-20250929"). If not specified, uses Haiku.'),
    statusMessage: z.string().optional().describe("Custom status message to display in spinner while hook runs")
});

export const HookSchema = z.discriminatedUnion("type", [CommandHookSchema, PromptHookSchema, AgentHookSchema]);

export const HookEntrySchema = z.object({
    matcher: z.string().optional().describe('String pattern to match (e.g. tool names like "Write")'),
    hooks: z.array(HookSchema).describe("List of hooks to execute when the matcher matches")
});

// Hook events enum (deobfuscated from $WA)
export const HookEvents = z.enum([
    "PreToolUse", "PostToolUse", "PostToolUseFailure", "Notification",
    "UserPromptSubmit", "SessionStart", "SessionEnd", "Stop",
    "SubagentStart", "SubagentStop", "PreCompact", "PermissionRequest"
]);

export const HooksConfigSchema = z.record(HookEvents, z.array(HookEntrySchema));

export const MarketplaceConfigSchema = z.object({
    source: MarketplaceSourceEnum.describe("Where to fetch the marketplace from"),
    installLocation: z.string().optional().describe("Local cache path where marketplace manifest is stored (auto-generated if not provided)")
});

export const McpServerAccessSchema = z.object({
    serverName: z.string().regex(/^[a-zA-Z0-9_-]+$/, "Server name can only contain letters, numbers, hyphens, and underscores").optional().describe("Name of the MCP server that users are allowed to configure"),
    serverCommand: z.array(z.string()).min(1, "Server command must have at least one element (the command)").optional().describe("Command array [command, ...args] to match exactly for allowed stdio servers"),
    serverUrl: z.string().optional().describe('URL pattern with wildcard support (e.g., "https://*.example.com/*") for allowed remote MCP servers')
}).refine(data => {
    return [data.serverName, data.serverCommand, data.serverUrl].filter(Boolean).length === 1;
}, { message: 'Entry must have exactly one of "serverName", "serverCommand", or "serverUrl"' });

export const SettingsSchema = z.object({
    $schema: z.string().optional().describe("JSON Schema reference for Claude Code settings"),
    apiKeyHelper: z.string().optional().describe("Path to a script that outputs authentication values"),
    awsCredentialExport: z.string().optional().describe("Path to a script that exports AWS credentials"),
    awsAuthRefresh: z.string().optional().describe("Path to a script that refreshes AWS authentication"),
    fileSuggestion: z.object({
        type: z.literal("command"),
        command: z.string()
    }).optional().describe("Custom file suggestion configuration for @ mentions"),
    cleanupPeriodDays: z.number().nonnegative().int().optional().describe("Number of days to retain chat transcripts (0 to disable cleanup)"),
    env: z.record(z.string(), z.string()).optional().describe("Environment variables to set for Claude Code sessions"),
    attribution: z.object({
        commit: z.string().optional().describe("Attribution text for git commits, including any trailers. Empty string hides attribution."),
        pr: z.string().optional().describe("Attribution text for pull request descriptions. Empty string hides attribution.")
    }).optional().describe("Customize attribution text for commits and PRs. Each field defaults to the standard Claude Code attribution if not set."),
    includeCoAuthoredBy: z.boolean().optional().describe("Deprecated: Use attribution instead. Whether to include Claude's co-authored by attribution in commits and PRs (defaults to true)"),
    permissions: PermissionsSettingsSchema.optional().describe("Tool usage permissions configuration"),
    model: z.string().optional().describe("Override the default model used by Claude Code"),
    enableAllProjectMcpServers: z.boolean().optional().describe("Whether to automatically approve all MCP servers in the project"),
    enabledMcpjsonServers: z.array(z.string()).optional().describe("List of approved MCP servers from .mcp.json"),
    disabledMcpjsonServers: z.array(z.string()).optional().describe("List of rejected MCP servers from .mcp.json"),
    allowedMcpServers: z.array(McpServerAccessSchema).optional().describe("Enterprise allowlist of MCP servers that can be used."),
    deniedMcpServers: z.array(McpServerAccessSchema).optional().describe("Enterprise denylist of MCP servers that are explicitly blocked."),
    hooks: HooksConfigSchema.optional().describe("Custom commands to run before/after tool executions"),
    disableAllHooks: z.boolean().optional().describe("Disable all hooks and statusLine execution"),
    allowManagedHooksOnly: z.boolean().optional().describe("When true, only hooks from managed settings run."),
    statusLine: z.object({
        type: z.literal("command"),
        command: z.string(),
        padding: z.number().optional()
    }).optional().describe("Custom status line display configuration"),
    enabledPlugins: z.record(z.string(), z.union([z.array(z.string()), z.boolean(), z.undefined()])).optional().describe('Enabled plugins using plugin-id@marketplace-id format.'),
    extraKnownMarketplaces: z.record(z.string(), MarketplaceConfigSchema).optional().describe("Additional marketplaces to make available for this repository."),
    skippedMarketplaces: z.array(z.string()).optional().describe("List of marketplace names the user has chosen not to install"),
    skippedPlugins: z.array(z.string()).optional().describe("List of plugin IDs the user has chosen not to install"),
    strictKnownMarketplaces: z.array(MarketplaceSourceEnum).optional().describe("Enterprise strict list of allowed marketplace sources."),
    blockedMarketplaces: z.array(MarketplaceSourceEnum).optional().describe("Enterprise blocklist of marketplace sources."),
    forceLoginMethod: z.enum(["claudeai", "console"]).optional().describe('Force a specific login method: "claudeai" or "console"'),
    forceLoginOrgUUID: z.string().optional().describe("Organization UUID to use for OAuth login"),
    otelHeadersHelper: z.string().optional().describe("Path to a script that outputs OpenTelemetry headers"),
    outputStyle: z.string().optional().describe("Controls the output style for assistant responses"),
    skipWebFetchPreflight: z.boolean().optional().describe("Skip the WebFetch blocklist check for enterprise environments"),
    sandbox: SandboxConfigSchema.optional(),
    spinnerTipsEnabled: z.boolean().optional().describe("Whether to show tips in the spinner"),
    syntaxHighlightingDisabled: z.boolean().optional().describe("Whether to disable syntax highlighting in diffs"),
    alwaysThinkingEnabled: z.boolean().optional().describe("When false, thinking is disabled."),
    promptSuggestionEnabled: z.boolean().optional().describe("When false, prompt suggestions are disabled."),
    agent: z.string().optional().describe("Name of an agent to use for the main thread."),
    companyAnnouncements: z.array(z.string()).optional().describe("Company announcements to display at startup"),
    pluginConfigs: z.record(z.string(), z.object({
        mcpServers: z.record(z.string(), z.record(z.string(), mcpUserConfigValueSchema())).optional()
    })).optional().describe("Per-plugin configuration including MCP server user configs"),
    remote: z.object({
        defaultEnvironmentId: z.string().optional().describe("Default environment ID to use for remote sessions")
    }).optional().describe("Remote session configuration"),
    autoUpdatesChannel: z.enum(["latest", "stable"]).optional().describe("Release channel for auto-updates (latest or stable)"),
    minimumVersion: z.string().optional().describe("Minimum version to stay on"),
    // Onboarding & Usage
    hasCompletedProjectOnboarding: z.boolean().optional().describe("Whether the user has completed the project onboarding flow"),
    projectOnboardingSeenCount: z.number().int().nonnegative().optional().describe("How many times the user has seen the project onboarding"),
    featureUsage: z.record(z.string(), z.number()).optional().describe("Track usage counts for various features")
}).passthrough();

function mcpUserConfigValueSchema() {
    return z.union([z.string(), z.number(), z.boolean(), z.array(z.string())]);
}
