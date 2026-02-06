/**
 * File: src/services/compaction/CompactionService.ts
 */

import { microcompactService } from './MicrocompactService.js';
import { Anthropic } from '../anthropic/AnthropicClient.js';
import { EnvService } from '../config/EnvService.js';
import { hookService } from '../hooks/HookService.js';
import { ModelResolver } from '../conversation/ModelResolver.js';

const AUTO_COMPACT_THRESHOLD = 40000; // uW2
const MIN_TOKENS = 10000; // DP1.minTokens
const MIN_TEXT_BLOCK_MESSAGES = 5; // DP1.minTextBlockMessages
const MAX_TOKENS = 40000; // DP1.maxTokens
const TOKEN_ESTIMATION_MULTIPLIER = 1.3333333333333333;

export interface CompactionResult {
    wasCompacted: boolean;
    summary?: string;
    compactedMessages?: any[];
    boundaryMarker?: any;
}

export class CompactionService {
    private autoCompactEnabled = true;

    constructor() { }

    async shouldAutoCompact(messages: any[]): Promise<boolean> {
        const autoCompactEnabled = !EnvService.get('DISABLE_COMPACT') && !EnvService.get('DISABLE_AUTO_COMPACT');
        if (!autoCompactEnabled) return false;

        const totalTokens = this.calculateTotalTokens(messages);
        return totalTokens > AUTO_COMPACT_THRESHOLD;
    }

    private calculateTotalTokens(messages: any[]): number {
        const contentSize = messages.reduce((acc, msg) => {
            return acc + this.estimateMessageTokens(msg);
        }, 0);
        return Math.ceil(contentSize * TOKEN_ESTIMATION_MULTIPLIER);
    }

    private estimateMessageTokens(message: any): number {
        const content = message.message?.content || message.content;
        return typeof content === 'string' ? content.length : JSON.stringify(content).length;
    }

    private isTextBlockMessage(message: any): boolean {
        // Corresponds to OqK in 2.1.19 - likely checks for user messages or non-tool assistant messages
        if (message.type === 'user') return true;
        if (message.type === 'assistant') {
            const content = message.message?.content;
            if (typeof content === 'string') return true;
            if (Array.isArray(content) && content.length > 0) {
                // Check if it has text content and not just tool use
                return content.some((block: any) => block.type === 'text');
            }
        }
        return false;
    }

    private getSplitIndex(messages: any[], startIndex?: number): number {
        if (messages.length === 0) return 0;

        const start = startIndex !== undefined && startIndex >= 0 ? startIndex + 1 : messages.length;
        let tokenCount = 0;
        let messageCount = 0;

        // Iterate backwards from the end (or start index)
        let splitIndex = start;

        for (let i = start - 1; i >= 0; i--) {
            const msg = messages[i];
            const tokens = Math.ceil(this.estimateMessageTokens(msg) * TOKEN_ESTIMATION_MULTIPLIER);

            tokenCount += tokens;
            if (this.isTextBlockMessage(msg)) {
                messageCount++;
            }

            splitIndex = i;

            if (tokenCount >= MAX_TOKENS) {
                break;
            }
            if (tokenCount >= MIN_TOKENS && messageCount >= MIN_TEXT_BLOCK_MESSAGES) {
                break;
            }
        }

        return this.adjustSplitForToolIntegrity(messages, splitIndex);
    }

    private adjustSplitForToolIntegrity(messages: any[], splitIndex: number): number {
        // Corresponds to clearConversation_50 in 2.1.19
        if (splitIndex <= 0 || splitIndex >= messages.length) return splitIndex;

        let finalSplit = splitIndex;

        // 1. Collect tool uses in the "kept" section (from splitIndex to end)
        // actually looking at clearConversation_50, it scans tool results in kept section?
        // Let's re-read carefully.
        // It collects filtered Y (tool results?) from K to end.
        // Then ensures their corresponding tool uses are also present.

        // Let's simplify: ensure no ToolResult is separated from its ToolUse.
        // Scan kept messages for ToolResults.
        const keptToolResultIds = new Set<string>();
        // Scan kept messages for ToolUses.
        const keptToolUseIds = new Set<string>();

        for (let i = finalSplit; i < messages.length; i++) {
            const msg = messages[i];
            const content = msg.message?.content || msg.content;
            if (Array.isArray(content)) {
                for (const block of content) {
                    if (block.type === 'tool_result') {
                        keptToolResultIds.add(block.toolUseId || block.tool_use_id);
                    }
                    if (block.type === 'tool_use') {
                        keptToolUseIds.add(block.id);
                    }
                }
            }
        }

        // Identify results that are missing their use in the kept section
        const missingUseIds = new Set<string>();
        for (const id of keptToolResultIds) {
            if (!keptToolUseIds.has(id)) {
                missingUseIds.add(id);
            }
        }

        // If we have missing uses, we must extend the kept section backwards to include them
        if (missingUseIds.size > 0) {
            for (let i = finalSplit - 1; i >= 0; i--) {
                if (missingUseIds.size === 0) break;

                const msg = messages[i];
                const content = msg.message?.content || msg.content;
                if (Array.isArray(content)) {
                    for (const block of content) {
                        if (block.type === 'tool_use' && missingUseIds.has(block.id)) {
                            finalSplit = i; // Move split point back
                            missingUseIds.delete(block.id);
                        }
                    }
                }
                // If we find the use, valid. If we find the result? well we are scanning backwards.
            }
        }

        return finalSplit;
    }

    async compact(messages: any[], customInstructions?: string): Promise<CompactionResult> {
        try {
            const preTokens = this.calculateTotalTokens(messages);
            // Hook: PreCompact
            await hookService.dispatch("PreCompact", {
                hook_event_name: "PreCompact",
                trigger: customInstructions ? "manual" : "auto",
                customInstructions: customInstructions ?? null
            });

            // 1. Generate summary using LLM
            const summary = await this.invokeSummarization(messages, customInstructions);

            // 2. Create compacted context
            const compactedMessages = [
                {
                    role: 'user',
                    content: summary,
                    isCompactSummary: true,
                    isVisibleInTranscriptOnly: true
                }
            ];

            // Boundary marker mimicking chunk1124 clearConversation_15
            const logicalParentUuid = messages[messages.length - 1]?.uuid;
            const boundaryMarker = {
                type: "system",
                subtype: "compact_boundary",
                content: "Conversation compacted",
                isMeta: false,
                timestamp: new Date().toISOString(),
                uuid: Math.random().toString(36).substring(7),
                level: "info",
                compactMetadata: {
                    trigger: customInstructions ? "manual" : "auto",
                    preTokens
                },
                ...(logicalParentUuid ? { logicalParentUuid } : {})
            };

            // Hook: PostCompact (or SessionStart as per chunk1074)
            await hookService.dispatch("SessionStart", {
                hook_event_name: "SessionStart",
                reason: "compact"
            });

            return {
                wasCompacted: true,
                summary,
                compactedMessages,
                boundaryMarker
            };
        } catch (error) {
            console.error("[CompactionService] Compaction failed:", error);
            return { wasCompacted: false };
        }
    }

    private getCompactionSystemPrompt(summary: string): string {
        return `This is a summary of the previous conversation context to stay within token limits:\n\n${summary}`;
    }

    private async invokeSummarization(messages: any[], instructions?: string): Promise<string> {
        const client = new Anthropic({
            baseUrl: EnvService.get("ANTHROPIC_BASE_URL")
        });

        const prompt = this.getSummarizationSystemPrompt(instructions);
        const model = ModelResolver.resolveModel(EnvService.get("CLAUDE_CODE_MODEL") || "claude-3-5-sonnet-20241022", false);

        const protoMessages = (Anthropic as any).prototype?.messages;
        const messagesApi = protoMessages?.create ? protoMessages : client.messages;

        const response = await messagesApi.create({
            model,
            max_tokens: 4096,
            system: prompt,
            messages: messages.map(m => ({
                role: m.role || m.type,
                content: m.message?.content || m.content
            }))
        });

        const content = response.data?.content;
        if (Array.isArray(content) && content[0]?.text) {
            return content[0].text;
        }

        throw new Error("Failed to generate summary: Invalid response format");
    }

    private getSummarizationSystemPrompt(additionalInstructions?: string): string {
        // Prompts from chunk640
        let prompt = `Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
  - Errors that you ran into and how you fixed them
  - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
6. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
7. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
8. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages: 
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.`;

        if (additionalInstructions) {
            prompt += `\n\nAdditional Instructions:\n${additionalInstructions}`;
        }
        return prompt;
    }

    async processTurn(messages: any[]): Promise<{ messages: any[], wasCompacted: boolean, boundaryMessage?: string }> {
        // First try microcompact
        const { messages: microcompacted, boundaryMessage } = await microcompactService.fetchContextData(messages);

        // Then check for autocompact
        if (await this.shouldAutoCompact(microcompacted)) {
            // Keep last 5 messages to maintain immediate context
            // TODO: Implement smarter splitting based on token blocks like 2.1.19
            // Smarter splitting based on token blocks like 2.1.19
            if (microcompacted.length > 0) {
                const splitIndex = this.getSplitIndex(microcompacted);
                const summarizeAll = splitIndex === 0 && microcompacted.length > 0;

                const messagesToSummarize = summarizeAll
                    ? microcompacted
                    : microcompacted.slice(0, splitIndex);
                const recentMessages = summarizeAll ? [] : microcompacted.slice(splitIndex);

                if (messagesToSummarize.length > 0) {
                    const result = await this.compact(messagesToSummarize);
                    if (result.wasCompacted && result.compactedMessages) {
                        // Combine: Summary + Boundary + Recent
                        const newHistory = [...result.compactedMessages];
                        if (result.boundaryMarker) {
                            newHistory.push(result.boundaryMarker);
                        }
                        newHistory.push(...recentMessages);

                        return {
                            messages: newHistory,
                            wasCompacted: true
                        };
                    }
                }
            }
        }

        return {
            messages: microcompacted,
            wasCompacted: !!boundaryMessage,
            boundaryMessage
        };
    }
}

export const compactionService = new CompactionService();
