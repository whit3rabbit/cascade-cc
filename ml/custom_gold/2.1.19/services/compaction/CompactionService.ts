/**
 * File: src/services/compaction/CompactionService.ts
 */

import { microcompactService } from './MicrocompactService.js';
import { Anthropic } from '../anthropic/AnthropicClient.js';
import { EnvService } from '../config/EnvService.js';
import { hookService } from '../hooks/HookService.js';
import { ModelResolver } from '../conversation/ModelResolver.js';

const WARNING_THRESHOLD = 20000; // BW2
const AUTO_COMPACT_THRESHOLD = 40000; // uW2
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
        // Multiplier from chunk1080/1079
        const contentSize = messages.reduce((acc, msg) => {
            const content = msg.message?.content || msg.content;
            return acc + (typeof content === 'string' ? content.length : JSON.stringify(content).length);
        }, 0);
        return Math.ceil(contentSize * TOKEN_ESTIMATION_MULTIPLIER);
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
                }
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

        const response = await client.messages.create({
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
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.`;

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
            const result = await this.compact(microcompacted);
            if (result.wasCompacted && result.compactedMessages) {
                // If we autocompact, we often discard the microcompact boundary in favor of the full compact boundary
                return {
                    messages: result.compactedMessages,
                    wasCompacted: true
                };
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
