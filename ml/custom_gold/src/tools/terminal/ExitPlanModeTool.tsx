
import React from 'react';
import { z } from 'zod';
import { Box, Text } from 'ink';

// Mocking these services as they are deobfuscated elsewhere
const getPlanFilePath = (agentId?: string) => `/path/to/plan-${agentId || 'current'}.md`;
const getPlanContent = (agentId?: string) => "Plan content...";

export const ExitPlanModeTool = {
    name: "ExitPlanMode",
    async description() {
        return "Prompts the user to exit plan mode and start coding";
    },
    async prompt() {
        return "Use this tool to indicate you have finished planning and are ready to start implementation. This will show your plan to the user for final approval.";
    },
    inputSchema: z.object({}).passthrough(),
    outputSchema: z.object({
        plan: z.string().nullable().describe("The plan that was presented to the user"),
        isAgent: z.boolean(),
        filePath: z.string().optional().describe("The file path where the plan was saved")
    }).passthrough(),

    userFacingName() { return ""; },
    isEnabled() { return true; },
    isConcurrencySafe() { return true; },
    isReadOnly() { return false; },
    requiresUserInteraction() { return true; },

    async checkPermissions(input: any) {
        return {
            behavior: "ask",
            message: "Exit plan mode?",
            updatedInput: input
        };
    },

    // render methods would refer to components in PlanStatus.tsx
    // ...

    async call(input: any, context: any) {
        const isAgent = !!context.agentId;
        const filePath = getPlanFilePath(context.agentId);
        const plan = getPlanContent(context.agentId);

        return {
            data: {
                plan,
                isAgent,
                filePath
            }
        };
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        const { isAgent, plan, filePath, awaitingLeaderApproval, requestId } = result;

        if (awaitingLeaderApproval) {
            return {
                type: "tool_result",
                tool_use_id: toolUseId,
                content: `Your plan has been submitted to the team lead for approval.\n\nPlan file: ${filePath}\n\nRequest ID: ${requestId}`
            };
        }

        if (isAgent) {
            return {
                type: "tool_result",
                tool_use_id: toolUseId,
                content: 'User has approved the plan. There is nothing else needed from you now. Please respond with "ok"'
            };
        }

        if (!plan || plan.trim() === "") {
            return {
                type: "tool_result",
                tool_use_id: toolUseId,
                content: "User has approved exiting plan mode. You can now proceed."
            };
        }

        return {
            type: "tool_result",
            tool_use_id: toolUseId,
            content: `User has approved your plan. You can now start coding. Start with updating your todo list if applicable\n\nYour plan has been saved to: ${filePath}\n\n## Approved Plan:\n${plan}`
        };
    }
};

export default ExitPlanModeTool;
