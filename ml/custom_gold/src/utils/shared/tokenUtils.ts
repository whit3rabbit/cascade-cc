/**
 * Token counting and estimation utilities.
 */

// Placeholder for external service calls
async function getClient(options: any): Promise<any> { return null; }
function getSelectedModel(): string { return "claude-3-5-sonnet-20241022"; }
function getBetaFlags(model: string): string[] { return []; }
function getProvider(): string { return "anthropic"; }
function getModelName(model: string): string { return model; }

export const DEFAULT_THINKING_BUDGET = 1024;
export const DEFAULT_MAX_TOKENS_THINKING = 2048;

/**
 * Checks if any message in the history contains a thinking block.
 */
export function hasThinkingBlock(messages: any[]): boolean {
    for (const msg of messages) {
        if (msg.role === "assistant" && Array.isArray(msg.content)) {
            for (const content of msg.content) {
                if (typeof content === "object" && content !== null &&
                    (content.type === "thinking" || content.type === "redacted_thinking")) {
                    return true;
                }
            }
        }
    }
    return false;
}

/**
 * Strips internal tool references from messages before sending to countTokens API
 * or as part of normalization.
 */
export function filterToolMessages(messages: any[]): any[] {
    return messages.map(msg => {
        if (!Array.isArray(msg.content)) return msg;
        const filteredContent = msg.content.map((content: any) => {
            if (content.type === "tool_use") {
                return {
                    type: "tool_use",
                    id: content.id,
                    name: content.name,
                    input: content.input
                };
            }
            if (content.type === "tool_result" && Array.isArray(content.content)) {
                // Assuming isToolReferenceBlock checks if content item is a tool reference
                const actualContent = content.content.filter((c: any) => c.type !== "tool_reference");
                if (actualContent.length === 0) {
                    return { ...content, content: [{ type: "text", text: "[tool references]" }] };
                }
                if (actualContent.length !== content.content.length) {
                    return { ...content, content: actualContent };
                }
            }
            return content;
        });
        return { ...msg, content: filteredContent };
    });
}

/**
 * Rough estimation of tokens based on character count (approx 4 chars per token).
 */
export function estimateTokens(text: string): number {
    return Math.round(text.length / 4);
}

/**
 * Counts tokens for a given prompt and set of tools using the Anthropic API.
 */
export async function countTokens(messages: any[], tools: any[]): Promise<number | null> {
    try {
        const model = getSelectedModel();
        const provider = getProvider();
        const betas = getBetaFlags(model);
        const containsThinking = hasThinkingBlock(messages);

        if (provider === "bedrock") {
            // Bedrock token count logic would go here
            return null;
        }

        const client = await getClient({ maxRetries: 1, model });
        if (!client) return null;

        const filteredMessages = filterToolMessages(messages);
        const response = await client.beta.messages.countTokens({
            model: getModelName(model),
            messages: filteredMessages.length > 0 ? filteredMessages : [{ role: "user", content: "foo" }],
            tools: tools.length > 0 ? tools : undefined,
            ...(betas.length > 0 ? { betas } : {}),
            ...(containsThinking ? {
                thinking: {
                    type: "enabled",
                    budget_tokens: DEFAULT_THINKING_BUDGET
                }
            } : {})
        });

        return response.input_tokens ?? null;
    } catch (error) {
        console.error("Error counting tokens:", error);
        return null;
    }
}

/**
 * Counts tokens by actually creating a tiny message and observing usage.
 * Used as a fallback or for more accurate counting including cache headers.
 */
export async function countTokensAccurate(messages: any[], tools: any[]): Promise<number | null> {
    try {
        const model = getSelectedModel(); // Or use a cheaper model if appropriate
        const client = await getClient({ maxRetries: 1, model });
        if (!client) return null;

        const containsThinking = hasThinkingBlock(messages);
        const filteredMessages = filterToolMessages(messages);
        const betas = getBetaFlags(model);

        const response = await client.beta.messages.create({
            model: getModelName(model),
            max_tokens: containsThinking ? DEFAULT_MAX_TOKENS_THINKING : 1,
            messages: filteredMessages.length > 0 ? filteredMessages : [{ role: "user", content: "count" }],
            tools: tools.length > 0 ? tools : undefined,
            ...(betas.length > 0 ? { betas } : {}),
            // metadata, etc...
            ...(containsThinking ? {
                thinking: {
                    type: "enabled",
                    budget_tokens: DEFAULT_THINKING_BUDGET
                }
            } : {})
        });

        const usage = response.usage;
        return (usage.input_tokens || 0) +
            (usage.cache_creation_input_tokens || 0) +
            (usage.cache_read_input_tokens || 0);
    } catch (error) {
        console.error("Error with accurate token count:", error);
        return null;
    }
}
