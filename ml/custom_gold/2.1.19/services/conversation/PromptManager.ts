/**
 * File: src/services/conversation/PromptManager.ts
 * Role: Logic for assembling the system prompt and managing token budget.
 */

import { getBaseConfigDir } from "../../utils/shared/runtimeAndEnv.js";
import { terminalLog } from "../../utils/shared/runtime.js";

export interface TokenBreakdown {
    categories: TokenCategory[];
    totalTokens: number;
    maxTokens: number;
    percentage: number;
    gridRows: any[][];
    model: string;
    // ... other metadata
}

export interface TokenCategory {
    name: string;
    tokens: number;
    color: string;
    isDeferred?: boolean;
}

/**
 * PromptManager service for assembling system prompts and tracking tokens.
 */
export class PromptManager {
    /**
     * Assembles the full system prompt based on user settings, agent definitions, and project context.
     * Corresponds to `wL` and `zH1` logic in the gold reference.
     */
    static async assembleSystemPrompt(options: any): Promise<string> {
        // This calls getBasePrompt (wL) and assembleAgentPrompt (zH1)
        const basePrompt = await this.getBasePrompt(options);
        const agentPrompt = await this.assembleAgentPrompt(options);

        let fullPrompt = `${basePrompt}\n\n${agentPrompt}`;

        if (options.planMode) {
            const planPrompt = `
Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits (with the exception of the plan file mentioned below), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received.

## Plan File Info:
A plan file exists at PLAN.md. You can read it and make incremental edits using the FileEdit tool.
You should build your plan incrementally by writing to or editing this file. NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.

## Plan Workflow

### Phase 1: Initial Understanding
Goal: Gain a comprehensive understanding of the user's request by reading through code and asking them questions.

1. Focus on understanding the user's request and the code associated with their request
2. Explore the codebase efficiently.

### Phase 2: Design
Goal: Design an implementation approach.
- Read critical files to deepen understanding.
- Ensure the plan aligns with the user's request.

### Phase 3: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Include only your recommended approach.
- Include verification steps.
- Typically use the "Ask" tool or "Exit Plan Mode" tool to confirm with user (for now, simply ask the user to confirm).

At the very end of your turn, once you have asked the user questions and are happy with your final plan file - you should ask the user to exit plan mode to proceed with implementation.
`;
            fullPrompt += `\n\n${planPrompt}`;
        }

        return fullPrompt;
    }

    /**
     * Gathers project-specific context (CLAUDE.md, git status, etc.)
     * Equivalent to `wL` in chunk1084 and implemented in chunk1452.
     */
    private static async getBasePrompt(options: any): Promise<string> {
        const parts = [
            "You are Claude Code, Anthropic's official CLI for Claude.",
            "You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.",
            "",
            "IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.",
            "",
            "# Tone and style",
            "- Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.",
            "- Your output will be displayed on a command line interface. Your responses should be short and concise.",
            "- Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks.",
            "- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one.",
            "",
            "# Professional objectivity",
            "Prioritize technical accuracy and truthfulness over validating the user's beliefs. Focus on facts and problem-solving, providing direct, objective technical info without any unnecessary superlatives, praise, or emotional validation.",
            "",
            "# No time estimates",
            "Never give time estimates or predictions for how long tasks will take. Focus on what needs to be done, not how long it might take.",
        ];

        // Add Logic for Tasks (mX tool) if present in options or default
        // For now, we assume standard tools are available.
        parts.push(`
# Task Management
You have access to the TaskManager tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
`);

        parts.push(`
# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- NEVER propose changes to code you haven't read. If a user asks about or wants you to modify a file, read it first. Understand existing code before suggesting modifications.
- Be careful not to introduce security vulnerabilities such as command injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities.
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary.
- Avoid backwards-compatibility hacks like renaming unused \`_vars\`, re-exporting types, adding \`// removed\` comments for removed code, etc. If something is unused, delete it completely.
`);

        parts.push(`
# Tool usage policy
- You can call multiple tools in a single response. If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel.
- If the user specifies that they want you to run tools "in parallel", you MUST send a single message with multiple tool use content blocks.
- Use specialized tools instead of bash commands when possible. For file operations, use dedicated tools: FileRead for reading files instead of cat/head/tail, FileEdit for editing instead of sed/awk.
`);

        return parts.join("\n");
    }

    /**
     * Customizes the prompt for a specific agent persona.
     * Equivalent to `zH1` in chunk1084.
     */
    private static async assembleAgentPrompt(options: any): Promise<string> {
        if (options.agentPrompt) return options.agentPrompt;

        // logic to load agent definitions from options.agents or config
        if (options.agent) {
            const agentDef = options.agents?.[options.agent];
            if (agentDef) {
                return `
# Role: ${options.agent}
${agentDef.description ? `Description: ${agentDef.description}` : ""}
${agentDef.prompt || ""}
`;
            }
        }
        return "";
    }

    /**
     * Compacts message history by summarizing older turns.
     */
    static async compactMessages(messages: any[], options: any): Promise<any[]> {
        if (messages.length < 10) return messages; // Don't compact small history

        const { Anthropic } = await import('../anthropic/AnthropicClient.js');
        const client = new Anthropic();

        const messagesToSummarize = messages.slice(0, -4); // Keep last 4 messages intact
        const recentMessages = messages.slice(-4);

        try {
            terminalLog("Compacting conversation history...", "info");
            const response = await client.messages.create({
                model: options.model || "claude-3-5-sonnet-20241022",
                max_tokens: 1500,
                system: `You are a helpful assistant specialized in summarizing technical conversations for an agent's memory.
Provide a concise but comprehensive summary including:
1. KEY FINDINGS: Facts discovered about the codebase (paths, variable values, logic).
2. ACTIONS TAKEN: Tools run and their outcomes.
3. CURRENT STATE: Active goals, pending tasks, and next steps.
4. CRITICAL MEMORY: Specific strings, IDs, or patterns that MUST be preserved for future tool calls.

The user will use this summary as context for future turns. DO NOT include meta-talk or filler sentences.`,
                messages: [
                    {
                        role: "user",
                        content: "Summarize the following conversation history:\n\n" +
                            messagesToSummarize.map(m => `${m.role.toUpperCase()}: ${typeof m.content === 'string' ? m.content : JSON.stringify(m.content)}`).join("\n\n")
                    }
                ]
            });

            const summary = response.data.content[0].text;

            return [
                {
                    role: "system",
                    content: `SUMMARY OF PREVIOUS CONVERSATION:\n${summary}`,
                    subtype: "compact_boundary",
                    isCompactSummary: true
                },
                ...recentMessages
            ];
        } catch (e) {
            terminalLog(`Failed to compact messages: ${e}`, "error");
            return messages; // Fallback to original
        }
    }

    private static tokenCache = new Map<string, TokenBreakdown>();

    /**
     * Calculates the token breakdown for the current session.
     * Equivalent to `clearConversation_83` in chunk1084.
     */
    static async getTokenBreakdown(options: any, messages: any[], systemPrompt: string): Promise<TokenBreakdown> {
        const cacheKey = JSON.stringify({ messages, systemPrompt, tools: options.tools?.map((t: any) => t.name) });
        if (this.tokenCache.has(cacheKey)) {
            return this.tokenCache.get(cacheKey)!;
        }

        // Local estimation (rough fallback to avoid latency)
        const estimateTokens = (text: string) => Math.ceil(text.length / 3.5);

        const localSystemTokens = estimateTokens(systemPrompt) + (options.tools?.length * 100 || 0);
        const localMessageTokens = messages.reduce((acc, m) => acc + estimateTokens(typeof m.content === 'string' ? m.content : JSON.stringify(m.content)), 0);
        const totalEstimate = localSystemTokens + localMessageTokens;

        const { Anthropic } = await import('../anthropic/AnthropicClient.js');
        const client = new Anthropic();

        try {
            // We only call the API if we really need a precise count, otherwise we use the estimate or a background refresh
            const response = await client.messages.countTokens({
                model: options.model || "claude-3-5-sonnet-20241022",
                system: systemPrompt,
                messages: messages.map(m => ({ role: m.role, content: m.content })),
                tools: options.tools?.map((t: any) => ({
                    name: t.name,
                    description: t.description,
                    input_schema: t.input_schema || t.parameters
                }))
            });

            const totalTokens = response.data.data.input_tokens || 0;
            const maxTokens = 200000;

            const result = {
                categories: [
                    { name: "System & Tools", tokens: response.data.data.system_tokens || 0, color: "promptBorder" },
                    { name: "Messages", tokens: response.data.data.message_tokens || 0, color: "purple" }
                ],
                totalTokens,
                maxTokens,
                percentage: Math.round((totalTokens / maxTokens) * 100),
                gridRows: [],
                model: options.model || "claude-3-5-sonnet-20241022"
            };

            this.tokenCache.set(cacheKey, result);
            return result;
        } catch (e) {
            terminalLog(`Failed to count tokens, using estimate: ${e}`, "warn");
            return {
                categories: [
                    { name: "System (est)", tokens: localSystemTokens, color: "promptBorder" },
                    { name: "Messages (est)", tokens: localMessageTokens, color: "purple" }
                ],
                totalTokens: totalEstimate,
                maxTokens: 200000,
                percentage: Math.round((totalEstimate / 200000) * 100),
                gridRows: [],
                model: options.model || "claude-3-5-sonnet-20241022"
            };
        }
    }

}
