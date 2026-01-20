import { z } from "zod";
import { Tool } from "./tool.js";

export const AgentInputSchema = z.object({
    description: z.string().describe("A short (3-5 word) description of the task"),
    prompt: z.string().describe("The task for the agent to perform"),
    subagent_type: z.string().describe("The type of specialized agent to use for this task"),
    model: z.enum(["sonnet", "opus", "haiku"]).optional().describe("Optional model to use for this agent. If not specified, inherits from parent. Prefer haiku for quick, straightforward tasks to minimize cost and latency."),
    resume: z.string().optional().describe("Optional agent ID to resume from. If provided, the agent will continue from the previous execution transcript."),
    run_in_background: z.boolean().optional().describe("Set to true to run this agent in the background. Use TaskOutput to read the output later.")
});

export const AgentTool: Tool = {
    name: "Agent",
    strict: true,
    input_examples: [{ prompt: "Investigate why the build is failing", context_files: ["src/index.ts"] }],
    description: async () => "Delegate a complex task to a sub-agent.",
    userFacingName: () => "Agent",
    getToolUseSummary: (input: any) => `Delegating to Agent: ${input.prompt}`,
    prompt: async () => "Delegate tasks to a sub-agent.",
    isEnabled: () => true,
    inputSchema: AgentInputSchema,
    outputSchema: z.object({ result: z.string() }),
    isConcurrencySafe: () => false,
    isReadOnly: () => false,
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: false }),
    async call(input: any) {
        // Stub implementation
        return {
            tool_use_id: "agent_stub",
            type: "tool_result",
            content: "Agent task completed (stub)."
        };
    },
    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: result.content
        };
    }
};
