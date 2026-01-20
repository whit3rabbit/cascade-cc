import { z } from "zod";
import fs from "fs/promises";
import path from "path";
import { Tool } from "./tool.js";

const TaskOutputInputSchema = z.object({
    task_id: z.string().describe("The task ID to get output from"),
    block: z.boolean().optional().describe("Whether to wait for completion"),
    timeout: z.number().optional().describe("Max wait time in ms")
});

const TaskOutputOutputSchema = z.object({
    output: z.string(),
    taskId: z.string(),
    completed: z.boolean().optional()
});

export const TaskOutputTool: Tool = {
    name: "TaskOutput",
    strict: true,
    input_examples: [{ taskId: "task_123" }],
    description: async () => "Read the output of a background task.",
    userFacingName: () => "Task Output",
    getToolUseSummary: (input: any) => `Reading output for task ${input.taskId}`,
    prompt: async () => "Read background task output.",
    isEnabled: () => true,
    inputSchema: TaskOutputInputSchema,
    outputSchema: TaskOutputOutputSchema,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: true }),

    async call(input: any) {
        const { taskId } = input;
        // Logic to read task output file. 
        // Assuming task outputs are stored in .claude/tasks/{taskId}.output based on BashTool.tsx
        const cwd = process.cwd();
        const outputPath = path.join(cwd, ".claude", "tasks", `${taskId}.output`);

        try {
            const output = await fs.readFile(outputPath, 'utf8');
            return {
                data: {
                    output,
                    taskId,
                    completed: false // Logic to determine if completed should be added if possible, but reading file essentially gives current state.
                }
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Error reading task output for ${taskId}: ${error.message}`
            };
        }
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        if (result.is_error) {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: result.content,
                is_error: true
            };
        }
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: result.data.output || "(No output yet)"
        };
    }
};
