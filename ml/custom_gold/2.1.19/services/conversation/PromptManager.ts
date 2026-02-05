/**
 * File: src/services/conversation/PromptManager.ts
 * Role: Logic for assembling the system prompt and managing token budget.
 */

import { getBaseConfigDir } from "../../utils/shared/runtimeAndEnv.js";
import { terminalLog } from "../../utils/shared/runtime.js";
import { AgentMessage, AgentMetadata } from "../../types/AgentTypes.js";

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
        return `
You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

# Tone and style
- Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.
- Your output will be displayed on a command line interface. Your responses should be short and concise. You can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
- Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like bash or code comments as means to communicate with the user during the session.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one. This includes markdown files.
- Do not use a colon before tool calls. Your tool calls may not be shown directly in the output, so text like "Let me read the file:" followed by a read tool call should just be "Let me read the file." with a period.

# Professional objectivity
Prioritize technical accuracy and truthfulness over validating the user's beliefs. Focus on facts and problem-solving, providing direct, objective technical info without any unnecessary superlatives, praise, or emotional validation. It is best for the user if Claude honestly applies the same rigorous standards to all ideas and disagrees when necessary, even if it may not be what the user wants to hear. Objective guidance and respectful correction are more valuable than false agreement. Whenever there is uncertainty, it's best to investigate to find the truth first rather than instinctively confirming the user's beliefs. Avoid using over-the-top validation or excessive praise when responding to users such as "You're absolutely right" or similar phrases.

# No time estimates
Never give time estimates or predictions for how long tasks will take, whether for your own work or for users planning their projects. Avoid phrases like "this will take me a few minutes," "should be done in about 5 minutes," "this is a quick fix," "this will take 2-3 weeks," or "we can do this later." Focus on what needs to be done, not how long it might take. Break work into actionable steps and let users judge timing for themselves.

# Task Management
You have access to the TaskManager tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.

# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- NEVER propose changes to code you haven't read. If a user asks about or wants you to modify a file, read it first. Understand existing code before suggesting modifications.
- Be careful not to introduce security vulnerabilities such as command injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities. If you notice that you wrote insecure code, immediately fix it.
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.
  - Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability. Don't add docstrings, comments, or type annotations to code you didn't change. Only add comments where the logic isn't self-evident.
  - Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use feature flags or backwards-compatibility shims when you can just change the code.
  - Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task—three similar lines of code is better than a premature abstraction.
- Avoid backwards-compatibility hacks like renaming unused \`_vars\`, re-exporting types, adding \`// removed\` comments for removed code, etc. If something is unused, delete it completely.

# Tool usage policy
- You can call multiple tools in a single response. If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel. Maximize use of parallel tool calls where possible to increase efficiency. However, if some tool calls depend on previous calls to inform dependent values, do NOT call these tools in parallel and instead call them sequentially. For instance, if one operation must complete before another starts, run these operations sequentially instead. Never use placeholders or guess missing parameters in tool calls.
- If the user specifies that they want you to run tools "in parallel", you MUST send a single message with multiple tool use content blocks.
- Use specialized tools instead of bash commands when possible, as this provides a better user experience. For file operations, use dedicated tools: FileRead for reading files instead of cat/head/tail, FileEdit for editing instead of sed/awk, and FileCreate for creating files instead of cat with heredoc or echo redirection. Reserve bash tools exclusively for actual system commands and terminal operations that require shell execution. NEVER use bash echo or other command-line tools to communicate thoughts, explanations, or instructions to the user. Output all communication directly in your response text instead.
- VERY IMPORTANT: When exploring the codebase to gather context or to answer a question that is not a needle query for a specific file/class/function, it is CRITICAL that you use the Search tool with subagent_type=researcher instead of running search commands directly.

# Code References
When referencing specific functions or pieces of code include the pattern \`file_path:line_number\` to allow the user to easily navigate to the source code location.
`;
    }

    /**
     * Customizes the prompt for a specific agent persona.
     * Equivalent to `zH1` in chunk1084.
     */
    private static async assembleAgentPrompt(options: any): Promise<string> {
        if (options.agentPrompt) return options.agentPrompt;

        const { CategoryRuleCache } = await import('./CategoryRuleCache.js');
        const { findAgent } = await import('../agents/AgentPersistence.js');

        const customInstructions = CategoryRuleCache.getInstance().getInstructions();

        let agentPrompt = "";
        let agentName = options.agent;

        if (agentName) {
            const agentDef = findAgent(agentName);
            if (agentDef) {
                agentPrompt = `
# Role: ${agentDef.name}
${agentDef.whenToUse ? `Description: ${agentDef.whenToUse}` : agentDef.description ? `Description: ${agentDef.description}` : ""}
${agentDef.systemPrompt || ""}
`;
            } else if (options.agents?.[agentName]) {
                // Fallback to inline agents from options
                const inlineDef = options.agents[agentName];
                agentPrompt = `
# Role: ${agentName}
${inlineDef.description ? `Description: ${inlineDef.description}` : ""}
${inlineDef.prompt || ""}
`;
            }
        }

        const sections = [agentPrompt, customInstructions].filter(Boolean);
        return sections.join("\n\n");
    }

    /**
     * Compacts message history by summarizing older turns.
     */
    /**
     * Compacts message history by summarizing older turns.
     */
    static async compactMessages(messages: AgentMessage[], options: any): Promise<AgentMessage[]> {
        const { compactionService } = await import('../compaction/CompactionService.js');

        try {
            // Note: CompactionService handles splitting recent messages vs summary context internally or via logic.
            // Actually CompactionService.compact takes ALL messages and summarizes them.
            // Wait, chunk1074 passes ALL messages (A).
            // But we might want to keep recent messages?
            // "chunk1074: let J = A.slice(H).filter(G => !AL(G));" where H is split point.
            // CompactionService.compact in 'src' seems to summarize everything passed to it.
            // If we want to keep recent messages, we should handle it here or in CompactionService.
            // However, CompactionService.compact implementation I read earlier (lines 44+) calls invokeSummarization on ALL messages passed.
            // And returns [User(Summary)] + [Boundary].
            // It seems it replaces EVERYTHING with summary.
            // If we want to keep recent messages (as typical RAG/compaction does), we should slice before passing?
            // "src/services/compaction/CompactionService.ts" logic:
            // "compactedMessages = [{ role: 'user', content: summary ... }]".
            // It does not append recent messages.
            // So I should pass ONLY the messages I want summarized?
            // BUT CompactionService calculates tokens on the messages passed.

            // Chunk1074 logic:
            // H = clearConversation_51(A, w) (Calculates split point based on tokens).
            // J = A.slice(H) (Keep messages from H onwards).
            // X = Summary of A[0..H-1].
            // Result = [Summary, ...J].

            // My CompactionService.compact does NOT do the splitting. It assumes input IS the history to compact.
            // So PromptManager should split.

            // Standard behavior: Keep last few turns.
            // In 2.1.19 `clearConversation_51` does smart splitting (min tokens, blocks, etc).
            // For now, I'll stick to a simple strategy: Keep last 4.

            if (messages.length < 10) return messages;

            const messagesToSummarize = messages.slice(0, -4);
            const recentMessages = messages.slice(-4);

            const result = await compactionService.compact(messagesToSummarize, "User manually requested compaction"); // Pass messagesToSummarize

            if (result.wasCompacted && result.compactedMessages) {
                const summaryMessages = [...result.compactedMessages];
                if (result.boundaryMarker) {
                    summaryMessages.unshift(result.boundaryMarker);
                }
                return [...summaryMessages, ...recentMessages] as AgentMessage[];
            }

            return messages;
        } catch (e) {
            terminalLog(`Failed to compact messages: ${e}`, "error");
            return messages;
        }
    }

    private static tokenCache = new Map<string, TokenBreakdown>();

    /**
     * Calculates the token breakdown for the current session.
     * Equivalent to `clearConversation_83` in chunk1084.
     */
    static async getTokenBreakdown(options: any, messages: AgentMessage[], systemPrompt: string): Promise<TokenBreakdown> {
        const cacheKey = JSON.stringify({ messages, systemPrompt, tools: options.tools?.map((t: any) => t.name) });
        if (this.tokenCache.has(cacheKey)) {
            return this.tokenCache.get(cacheKey)!;
        }

        // Local estimation (rough fallback to avoid latency)
        // Aligned with chunk1080: Math.ceil(charCount * 1.3333333333333333)
        const estimateTokens = (text: string) => Math.ceil(text.length * 1.3333333333333333);

        const localSystemTokens = estimateTokens(systemPrompt) + (options.tools?.length * 100 || 0);
        const localMessageTokens = messages.reduce((acc, m) => acc + estimateTokens(typeof m.content === 'string' ? m.content : JSON.stringify(m.content)), 0);
        const totalEstimate = localSystemTokens + localMessageTokens;

        const { Anthropic } = await import('../anthropic/AnthropicClient.js');
        const client = new Anthropic();

        try {
            // We only call the API if we really need a precise count or if it's the first time
            const response = await client.messages.countTokens({
                model: options.model || "claude-3-5-sonnet-20241022",
                system: systemPrompt,
                messages: messages.map(m => ({ role: m.role || 'user', content: m.content || m.message })),
                tools: options.tools?.map((t: any) => ({
                    name: t.name,
                    description: t.description,
                    input_schema: t.input_schema || t.parameters
                }))
            });

            const totalTokens = response.input_tokens || (response as any).data?.input_tokens || 0;
            const maxTokens = 200000;

            const result: TokenBreakdown = {
                categories: [
                    { name: "System & Tools", tokens: (response as any).system_tokens || 0, color: "promptBorder" },
                    { name: "Messages", tokens: (response as any).message_tokens || 0, color: "purple" }
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
