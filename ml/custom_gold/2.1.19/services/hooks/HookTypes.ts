import { z } from 'zod';

export const HookEventSchema = z.enum([
    'SessionStart',
    'UserPromptSubmit',
    'PreToolUse',
    'PostToolUse',
    'PostToolUseFailure',
    'Notification',
    'PermissionRequest',
    // 'Setup', // Noted in deobfuscated code but typically less common, including for completeness
    // 'SubagentStart' // Ditto
]);

export type HookEvent = z.infer<typeof HookEventSchema>;

export const HookPermissionDecisionSchema = z.enum(['allow', 'deny', 'ask']);
export type HookPermissionDecision = z.infer<typeof HookPermissionDecisionSchema>;

// Input passed TO the hook (JSON input)
export const HookInputSchema = z.object({
    hook_event_name: HookEventSchema,
    // Common fields typically present in all inputs
    session_id: z.string().optional(),
    transcript_path: z.string().optional(),
    cwd: z.string().optional(),
    permission_mode: z.string().optional(), // 'prompt-always' | 'prompt-sensitive' etc.

    // Specific fields per event
    // PreToolUse
    tool_name: z.string().optional(),
    tool_input: z.record(z.string(), z.unknown()).optional(),
    tool_use_id: z.string().optional(),

    // UserPromptSubmit
    prompt: z.string().optional(),

    // PostToolUse / Failure
    tool_response: z.unknown().optional(), // For PostToolUse
    error: z.unknown().optional(), // For PostToolUseFailure
    is_interrupt: z.boolean().optional(),

    // Notification
    message: z.string().optional(),
    title: z.string().optional(),
    notification_type: z.string().optional(),

    // PermissionRequest
    // Use schema from chunk1154 implementation if possible for more specific fields
});

export type HookInput = z.infer<typeof HookInputSchema>;

// Output received FROM the hook (JSON output)
export const HookSpecificOutputSchema = z.union([
    z.object({
        hookEventName: z.literal('PreToolUse'),
        permissionDecision: HookPermissionDecisionSchema.optional(),
        permissionDecisionReason: z.string().optional(),
        updatedInput: z.record(z.string(), z.unknown()).optional(),
        additionalContext: z.string().optional(),
    }),
    z.object({
        hookEventName: z.literal('UserPromptSubmit'),
        additionalContext: z.string().optional(),
        // updatedPrompt? Not strictly in reference but useful. Reference shows 'additionalContext' (required) in chunk1160?
        // chunk1160 says: "for UserPromptSubmit": { hookEventName: '"UserPromptSubmit"', additionalContext: "string (required)" }
    }),
    z.object({
        hookEventName: z.literal('SessionStart'),
        additionalContext: z.string().optional(),
    }),
    z.object({
        hookEventName: z.literal('PostToolUse'),
        additionalContext: z.string().optional(),
        updatedMCPToolOutput: z.unknown().describe("Updates the output for MCP tools").optional()
    }),
    z.object({
        hookEventName: z.literal('PostToolUseFailure'),
        additionalContext: z.string().optional(),
    }),
    z.object({
        hookEventName: z.literal('Notification'),
        additionalContext: z.string().optional(),
    }),
    z.object({
        hookEventName: z.literal('PermissionRequest'),
        decision: z.union([
            z.object({
                behavior: z.literal('allow'),
                updatedInput: z.record(z.string(), z.unknown()).optional(),
                // updatedPermissions: z.array(permissionRuleSchema).optional() // Omitting complex recursive schema for now unless needed
            }),
            z.object({
                behavior: z.literal('deny'),
                message: z.string().optional(),
                interrupt: z.boolean().optional(),
            })
        ]).optional()
    })
]);

export const HookOutputSchema = z.object({
    continue: z.boolean().describe("Whether Claude should continue after hook (default: true)").optional(),
    suppressOutput: z.boolean().describe("Hide stdout from transcript (default: false)").optional(),
    stopReason: z.string().describe("Message shown when continue is false").optional(),
    decision: z.enum(["approve", "block"]).optional(), // Simple approval decision
    reason: z.string().describe("Explanation for the decision").optional(),
    systemMessage: z.string().describe("Warning message shown to the user").optional(),

    hookSpecificOutput: HookSpecificOutputSchema.optional(),
});

export type HookOutput = z.infer<typeof HookOutputSchema>;

// The configuration object in settings.json
// hooks: { [eventName]: HookConfigEntry[] }
export const HookConfigEntrySchema = z.object({
    commands: z.union([z.string(), z.array(z.string())]).optional(),
    command: z.string().optional(),
    cwd: z.string().optional(),
    timeout: z.number().optional(),
});

export type HookConfigEntry = z.infer<typeof HookConfigEntrySchema>;
// Mapping of event name to list of hooks
export type HooksConfig = Partial<Record<HookEvent, HookConfigEntry[]>>;
