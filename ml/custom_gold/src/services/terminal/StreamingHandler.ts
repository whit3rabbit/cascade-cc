
import { randomUUID } from "node:crypto";
import { log } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { createAssistantMessage, normalizeMessages } from "./MessageFactory.js";
import { getStreamingContent } from "../claude/claudeApi.js";
import { calculateTokenStats } from "../history/ConversationHistoryManager.js";
import { getGlobalState } from "../session/globalState.js";

const logger = log("streaming");

/**
 * Wraps text in a reminder tag. (H5)
 */
export function wrapInReminder(text: string): string {
    return `<reminder>\n${text}\n</reminder>`;
}

/**
 * Handles the streaming response from the API, builds the full message, and manages events.
 * Based on chunk_580.ts (zHA)
 */
export async function* getCompletionStream(params: {
    messages: any[];
    systemPrompt: string;
    maxThinkingTokens?: number;
    tools: any[];
    signal: AbortSignal;
    options: any;
}): AsyncGenerator<any> {
    const {
        messages,
        systemPrompt,
        maxThinkingTokens,
        tools,
        signal,
        options
    } = params;

    // Yield start event for MainLoop
    yield { type: "stream_request_start" };

    try {
        // Prepare tool schemas (simplified from HX1/SW9)
        const toolSchemas = tools.map((t: any) => {
            if ("inputJSONSchema" in t) return t.inputJSONSchema;
            return {
                name: t.name,
                description: t.description,
                input_schema: t.inputSchema
            };
        });

        // Normalize messages (GJ/normalizeMessages)
        const normalizedMessages = normalizeMessages(messages);

        // Remove 'id' from user messages if present before sending to API (API may reject unknown fields)
        const apiMessages = normalizedMessages.map(m => {
            const hasId = m.message.role === 'user' && m.message.id;
            if (hasId) {
                const { id, ...rest } = m.message;
                return rest;
            }
            return m.message;
        });

        // Prepare request parameters
        const requestParams: any = {
            messages: apiMessages,
            systemPrompt,
            model: options.model,
            tools: toolSchemas.length > 0 ? toolSchemas : undefined,
            max_tokens: options.maxOutputTokensOverride || 4096,
            temperature: options.temperatureOverride ?? 1,
            signal
        };

        if (maxThinkingTokens && maxThinkingTokens > 0) {
            requestParams.thinking = { type: "enabled", budget_tokens: maxThinkingTokens };
        }

        // Call API
        const stream = getStreamingContent(requestParams);

        let currentMessage: any = null;
        let contentBlocks: any[] = [];
        let usage = { input_tokens: 0, output_tokens: 0 };
        const thinkingBuffer: string[] = [];

        for await (const event of stream) {

            // Yield raw events if needed by UI
            yield { type: "stream_event", event };

            switch (event.type) {
                case "message_start":
                    currentMessage = { ...event.message, content: [] };
                    usage = event.message.usage || usage;
                    break;
                case "content_block_start":
                    contentBlocks[event.index] = { ...event.content_block };
                    if (event.content_block.type === "tool_use") {
                        contentBlocks[event.index].input = "";
                    } else if (event.content_block.type === "text") {
                        contentBlocks[event.index].text = "";
                    } else if (event.content_block.type === "thinking") {
                        contentBlocks[event.index].thinking = "";
                    }
                    break;
                case "content_block_delta":
                    const block = contentBlocks[event.index];
                    if (event.delta.type === "text_delta") {
                        block.text += event.delta.text;
                    } else if (event.delta.type === "input_json_delta") {
                        block.input += event.delta.partial_json;
                    } else if (event.delta.type === "thinking_delta") {
                        block.thinking += event.delta.thinking;
                    }
                    break;
                case "content_block_stop":
                    // Parse tool inputs when block stops
                    const stoppedBlock = contentBlocks[event.index];
                    if (stoppedBlock.type === "tool_use") {
                        try {
                            if (typeof stoppedBlock.input === 'string') {
                                stoppedBlock.input = JSON.parse(stoppedBlock.input);
                            }
                        } catch (e) {
                            logger.error("Failed to parse tool input JSON", e);
                        }
                    }
                    break;
                case "message_delta":
                    if (event.usage) {
                        usage.output_tokens = event.usage.output_tokens;
                    }
                    if (event.delta && event.delta.stop_reason) {
                        currentMessage.stop_reason = event.delta.stop_reason;
                    }
                    if (event.delta && event.delta.stop_sequence) {
                        currentMessage.stop_sequence = event.delta.stop_sequence;
                    }
                    break;
                case "message_stop":
                    break;
            }
        }

        if (currentMessage) {
            // Finalize message
            currentMessage.content = contentBlocks;
            currentMessage.usage = usage;

            const fullMessage = createAssistantMessage(contentBlocks, {
                uuid: randomUUID(),
                model: options.model,
                usage,
                requestId: currentMessage.id, // Or from header if available
                timestamp: new Date().toISOString()
            });

            // Update fullMessage internals to match what MainLoop expects
            fullMessage.message = currentMessage; // Ensure exact structure matches API response structure for content

            yield fullMessage;
        }

    } catch (err: any) {
        logger.error("Error in completion stream", err);
        // MainLoop handles errors thrown from here (e.g. falling back to non-streaming if needed, but we implement just streaming here)
        throw err;
    }
}
