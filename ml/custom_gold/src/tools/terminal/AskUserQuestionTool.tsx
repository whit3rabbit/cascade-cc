
import React from 'react';
import { z } from 'zod';
import { Box, Text } from 'ink';

const OptionSchema = z.object({
    label: z.string().describe("The display text for this option that the user will see and select. Should be concise (1-5 words) and clearly describe the choice."),
    description: z.string().describe("Explanation of what this option means or what will happen if chosen. Useful for providing context about trade-offs or implications.")
});

const QuestionSchema = z.object({
    question: z.string().describe("The complete question to ask the user. Should be clear, specific, and end with a question mark. Example: \"Which library should we use for date formatting?\" If multiSelect is true, phrase it accordingly, e.g. \"Which features do you want to enable?\""),
    header: z.string().describe("Very short label displayed as a chip/tag (max 12 chars). Examples: \"Auth method\", \"Library\", \"Approach\"."),
    options: z.union([
        z.tuple([OptionSchema, OptionSchema]),
        z.tuple([OptionSchema, OptionSchema, OptionSchema]),
        z.tuple([OptionSchema, OptionSchema, OptionSchema, OptionSchema])
    ]).describe("The available choices for this question. Must have 2-4 options. Each option should be a distinct, mutually exclusive choice (unless multiSelect is enabled). There should be no 'Other' option, that will be provided automatically."),
    multiSelect: z.boolean().describe("Set to true to allow the user to select multiple options instead of just one. Use when choices are not mutually exclusive.")
});

// AskUserQuestion (BW1)
export const AskUserQuestionTool = {
    name: "AskUserQuestion",
    async description() {
        return "Asks the user one or more multiple-choice questions to clarify intent or approach.";
    },
    async prompt() {
        return "Use this tool when you need to clarify the user's requirements or get approval for a specific approach among multiple valid alternatives. This is better than AskUserQuestion (plain text) when there are clear distinct paths to choose from.";
    },
    inputSchema: z.object({
        questions: z.union([
            z.tuple([QuestionSchema]),
            z.tuple([QuestionSchema, QuestionSchema]),
            z.tuple([QuestionSchema, QuestionSchema, QuestionSchema]),
            z.tuple([QuestionSchema, QuestionSchema, QuestionSchema, QuestionSchema])
        ]).describe("Questions to ask the user (1-4 questions)"),
        answers: z.record(z.string(), z.string()).optional().describe("User answers collected by the permission component")
    }),
    outputSchema: z.object({
        questions: z.array(QuestionSchema),
        answers: z.record(z.string(), z.string())
    }),

    isEnabled() { return true; },
    isConcurrencySafe() { return true; },
    isReadOnly() { return true; },
    requiresUserInteraction() { return true; },

    async checkPermissions(input: any) {
        return {
            behavior: "ask",
            message: "Answer questions?",
            updatedInput: input
        };
    },

    async call(input: any) {
        return {
            data: {
                questions: input.questions,
                answers: input.answers || {}
            }
        };
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        const { answers } = result;
        const answerStr = Object.entries(answers).map(([q, a]) => `"${q}"="${a}"`).join(", ");
        return {
            type: "tool_result",
            tool_use_id: toolUseId,
            content: `User has answered your questions: ${answerStr}. You can now continue with the user's answers in mind.`
        };
    }
};

export default AskUserQuestionTool;
