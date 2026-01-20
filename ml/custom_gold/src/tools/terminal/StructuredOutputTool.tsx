
import React from 'react';
import { z } from 'zod';
import { Box, Text } from 'ink';

// Note: In reality this tool is dynamically constructed with a specific JSON schema.
// This is the base tool implementation.

export const StructuredOutputTool = {
    name: "StructuredOutput",
    description: "Return structured output in the requested format",
    async prompt() {
        return "Use this tool to return your final response in the requested structured format. You MUST call this tool exactly once at the end of your response to provide the structured output.";
    },
    inputSchema: z.record(z.any()).describe("Structured output tool result"),
    outputSchema: z.string(),
    isEnabled() { return true; },
    isConcurrencySafe() { return true; },
    isReadOnly() { return true; },
    isDestructive() { return false; },
    isOpenWorld() { return false; },

    async call(input: any) {
        // The actual validation against inputJSONSchema happens in the tool runner/wrapper.
        return {
            data: "Structured output provided successfully",
            structured_output: input
        };
    },

    async checkPermissions(input: any) {
        return {
            behavior: "allow",
            updatedInput: input
        };
    },

    renderToolUseMessage(input: any) {
        const keys = Object.keys(input);
        if (keys.length === 0) return null;
        if (keys.length <= 3) return keys.map(k => `${k}: ${JSON.stringify(input[k])}`).join(", ");
        return `${keys.length} fields: ${keys.slice(0, 3).join(", ")}…`;
    },

    userFacingName() { return "StructuredOutput"; },
    renderToolUseRejectedMessage() { return "Structured output rejected"; },
    renderToolUseErrorMessage() { return "Structured output error"; },
    renderToolUseProgressMessage() { return null; },
    renderToolResultMessage(result: any) { return result; },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: result
        };
    }
};

export default StructuredOutputTool;
