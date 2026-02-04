/**
 * File: src/services/terminal/schemas.ts
 * Role: UTILITY_HELPER
 * Zod schemas for terminal services.
 */

import { z } from 'zod';

export const ToolPermissionResponseSchema = z.union([
    z.object({
        behavior: z.literal("allow"),
        updatedInput: z.record(z.string(), z.unknown()).optional(),
        updatedPermissions: z.array(z.any()).optional()
    }),
    z.object({
        behavior: z.literal("deny"),
        message: z.string().optional(),
        interrupt: z.boolean().optional()
    })
]);

const AsyncHookResult = z.object({
    async: z.literal(true),
    asyncTimeout: z.number().optional()
});

const SyncHookResult = z.object({
    continue: z.boolean().describe("Whether Claude should continue after hook (default: true)").optional(),
    suppressOutput: z.boolean().describe("Hide stdout from transcript (default: false)").optional(),
    stopReason: z.string().describe("Message shown when continue is false").optional(),
    decision: z.enum(["approve", "block"]).optional(),
    reason: z.string().describe("Explanation for the decision").optional(),
    systemMessage: z.string().describe("Warning message shown to the user").optional(),
    hookSpecificOutput: z.union([
        z.object({
            hookEventName: z.literal("PreToolUse"),
            permissionDecision: z.enum(["allow", "deny", "ask"]).optional(),
            permissionDecisionReason: z.string().optional(),
            updatedInput: z.record(z.string(), z.unknown()).optional(),
            additionalContext: z.string().optional()
        }),
        // Add other cases as needed...
        z.any()
    ]).optional()
});

export const HookCallbackSchema = z.union([AsyncHookResult, SyncHookResult]);

// Aliases for compatibility
export const uv1 = ToolPermissionResponseSchema;
export const PV1 = HookCallbackSchema;

export const LspToolInputSchema = z.strictObject({
    operation: z.enum(['goToDefinition', 'findReferences', 'hover', 'documentSymbol', 'workspaceSymbol', 'goToImplementation', 'prepareCallHierarchy', 'incomingCalls', 'outgoingCalls']).describe('The LSP operation to perform'),
    filePath: z.string().describe('The absolute or relative path to the file'),
    lineNumber: z.number().int().positive().describe('The line number (1-based, as shown in editors)'),
    character: z.number().int().positive().describe('The character offset (1-based, as shown in editors)')
});

export const LspToolOutputSchema = z.object({
    operation: z.enum(['goToDefinition', 'findReferences', 'hover', 'documentSymbol', 'workspaceSymbol', 'goToImplementation', 'prepareCallHierarchy', 'incomingCalls', 'outgoingCalls']).describe('The LSP operation that was performed'),
    result: z.string().describe('The formatted result of the LSP operation'),
    filePath: z.string().describe('The file path the operation was performed on'),
    resultCount: z.number().int().nonnegative().optional().describe('Number of results (definitions, references, symbols)'),
    fileCount: z.number().int().nonnegative().optional().describe('Number of files containing results')
});

export type LspOperation = z.infer<typeof LspToolInputSchema>;
export type LspToolOutput = z.infer<typeof LspToolOutputSchema>;
// Compatibility alias if needed
export type LspToolResult = LspToolOutput;
