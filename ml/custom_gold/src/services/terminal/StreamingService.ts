
// Logic from chunk_580.ts (Claude API Streaming & Error Handling)

/**
 * Interface representing content blocks in a message.
 */
export interface ContentBlock {
    type: "text" | "tool_use" | "server_tool_use" | "thinking";
    text?: string;
    input?: string; // Partial JSON for tool calls
    thinking?: string;
    signature?: string;
    [key: string]: any;
}

/**
 * Handles Claude API stream events and updates the message state.
 */
export async function* handleClaudeStream(stream: AsyncIterable<any>, options: any) {
    let contentBlocks: ContentBlock[] = [];
    let usage = { input_tokens: 0, output_tokens: 0 };
    let stopReason: string | null = null;
    let startTime = Date.now();
    let ttft: number | null = null; // Time to First Token

    try {
        for await (const event of stream) {
            switch (event.type) {
                case "message_start":
                    // Initialize message state
                    usage = event.message.usage;
                    break;

                case "content_block_start":
                    const newBlock: ContentBlock = {
                        type: event.content_block.type,
                        ...event.content_block
                    };
                    contentBlocks[event.index] = newBlock;
                    break;

                case "content_block_delta":
                    const block = contentBlocks[event.index];
                    if (!block) break;

                    if (ttft === null) ttft = Date.now() - startTime;

                    switch (event.delta.type) {
                        case "text_delta":
                            if (block.type === "text") block.text += event.delta.text;
                            break;
                        case "input_json_delta":
                            if (block.type === "tool_use" || block.type === "server_tool_use") {
                                block.input = (block.input || "") + event.delta.partial_json;
                            }
                            break;
                        case "thinking_delta":
                            if (block.type === "thinking") block.thinking += event.delta.thinking;
                            break;
                        case "signature_delta":
                            if (block.type === "thinking") block.signature = event.delta.signature;
                            break;
                    }
                    break;

                case "message_delta":
                    // Update usage and check stop reason
                    if (event.usage) {
                        usage.output_tokens = event.usage.output_tokens;
                    }
                    stopReason = event.delta.stop_reason;

                    if (stopReason === "max_tokens") {
                        yield {
                            type: "warning",
                            content: "Claude's response exceeded the output token maximum."
                        };
                    }
                    break;

                case "message_stop":
                    // Stream finished successfully
                    break;
            }

            yield {
                type: "stream_event",
                event,
                currentContent: contentBlocks,
                usage
            };
        }
    } catch (error: any) {
        console.error("Streaming error, falling back to non-streaming:", error);
        // Fallback logic would go here in the real implementation
        throw error;
    }
}

/**
 * Safely aborts a request.
 */
export function safelyAbortRequest(request: any) {
    if (!request) return;
    try {
        if (!request.ended && !request.aborted) {
            request.abort();
        }
    } catch (err) {
        // Ignore errors during abort
    }
}
