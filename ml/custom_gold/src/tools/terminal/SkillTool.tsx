
import React from 'react';
import { z } from 'zod';
import { Box, Text } from 'ink';

// SkillTool (sr)
export const SkillTool = {
    name: "Skill",
    inputSchema: z.object({
        skill: z.string().describe('The skill name. E.g., "commit", "review-pr", or "pdf"'),
        args: z.string().optional().describe("Optional arguments for the skill")
    }),
    outputSchema: z.object({
        success: z.boolean().describe("Whether the skill is valid"),
        commandName: z.string().describe("The name of the skill"),
        allowedTools: z.array(z.string()).optional().describe("Tools allowed by this skill"),
        model: z.string().optional().describe("Model override if specified")
    }),

    async description({ skill }: { skill: string }) {
        return `Execute skill: ${skill}`;
    },

    async prompt() {
        return "Use this tool to execute a specific skill or slash command. This allows you to leverage specialized workflows for tasks like committing code, reviewing pull requests, or processing specific file types.";
    },

    userFacingName(input: any) {
        if (input?.skill) return `/${input.skill}`;
        return "Skill";
    },

    isConcurrencySafe() { return false; },
    isEnabled() { return true; },
    isReadOnly() { return false; },

    async validateInput({ skill }: { skill: string }) {
        const trimmed = skill.trim();
        if (!trimmed) return { result: false, message: `Invalid skill format: ${skill}`, errorCode: 1 };

        const name = trimmed.startsWith('/') ? trimmed.substring(1) : trimmed;
        // Logic to check if skill exists in registry
        return { result: true };
    },

    async checkPermissions({ skill }: { skill: string }, context: any) {
        const name = skill.trim().startsWith('/') ? skill.trim().substring(1) : skill.trim();

        // Check permission rules for this skill
        // ... (Decision logic from chunk_488:374)

        return {
            behavior: "ask",
            message: `Execute skill: ${name}`,
            metadata: {
                // ...
            }
        };
    },

    async call({ skill, args }: { skill: string, args?: string }, context: any) {
        const name = skill.trim().startsWith('/') ? skill.trim().substring(1) : skill.trim();

        // Logic to dispatch the slash command
        // ... (Exec logic from chunk_488:427)

        return {
            data: {
                success: true,
                commandName: name,
                // allowedTools, model, etc.
            },
            // newMessages, contextModifier...
        };
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        return {
            type: "tool_result",
            tool_use_id: toolUseId,
            content: `Launching skill: ${result.commandName}`
        };
    }
};

export default SkillTool;
