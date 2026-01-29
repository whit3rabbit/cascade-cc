/**
 * File: src/utils/shared/conversationUtils.ts
 * Role: Utilities for managing conversation messages, token usage, and response generation.
 */

// --- Types ---
export interface TokenUsage {
    inputTokens: number;
    outputTokens: number;
    cacheCreationInputTokens: number;
    cacheReadInputTokens: number;
    server_tool_use: {
        web_search_requests: number;
        web_fetch_requests: number;
    };
    service_tier: string;
    cache_creation: {
        ephemeral_1h_input_tokens: number;
        ephemeral_5m_input_tokens: number;
    };
}

export interface Message {
    type: string;
    text?: string;
    content?: any;
    cacheScope?: string | null;
}

/**
 * Updates token usage metrics for a conversation by merging new updates.
 */
export function updateTokenUsage(current: TokenUsage, update: Partial<TokenUsage>): TokenUsage {
    return {
        inputTokens: (update.inputTokens && update.inputTokens > 0) ? update.inputTokens : current.inputTokens,
        cacheCreationInputTokens: (update.cacheCreationInputTokens && update.cacheCreationInputTokens > 0) ? update.cacheCreationInputTokens : current.cacheCreationInputTokens,
        cacheReadInputTokens: (update.cacheReadInputTokens && update.cacheReadInputTokens > 0) ? update.cacheReadInputTokens : current.cacheReadInputTokens,
        outputTokens: update.outputTokens ?? current.outputTokens,
        server_tool_use: {
            web_search_requests: update.server_tool_use?.web_search_requests ?? current.server_tool_use.web_search_requests,
            web_fetch_requests: update.server_tool_use?.web_fetch_requests ?? current.server_tool_use.web_fetch_requests
        },
        service_tier: current.service_tier,
        cache_creation: {
            ephemeral_1h_input_tokens: update.cache_creation?.ephemeral_1h_input_tokens ?? current.cache_creation.ephemeral_1h_input_tokens,
            ephemeral_5m_input_tokens: update.cache_creation?.ephemeral_5m_input_tokens ?? current.cache_creation.ephemeral_5m_input_tokens
        }
    };
}

/**
 * Combines token usage metrics from two separate measurement points.
 */
export function combineTokenUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
    return {
        inputTokens: a.inputTokens + b.inputTokens,
        cacheCreationInputTokens: a.cacheCreationInputTokens + b.cacheCreationInputTokens,
        cacheReadInputTokens: a.cacheReadInputTokens + b.cacheReadInputTokens,
        outputTokens: a.outputTokens + b.outputTokens,
        server_tool_use: {
            web_search_requests: a.server_tool_use.web_search_requests + b.server_tool_use.web_search_requests,
            web_fetch_requests: a.server_tool_use.web_fetch_requests + b.server_tool_use.web_fetch_requests
        },
        service_tier: b.service_tier,
        cache_creation: {
            ephemeral_1h_input_tokens: a.cache_creation.ephemeral_1h_input_tokens + b.cache_creation.ephemeral_1h_input_tokens,
            ephemeral_5m_input_tokens: a.cache_creation.ephemeral_5m_input_tokens + b.cache_creation.ephemeral_5m_input_tokens
        }
    };
}

/**
 * Adjusts the maximum tokens allowed for a conversation, considering the thinking budget.
 */
export function adjustMaxTokens(conversationSettings: any, maxOutputTokens: number): any {
    const maxTokens = Math.min(conversationSettings.max_tokens, maxOutputTokens);
    const updatedSettings = { ...conversationSettings };

    if (updatedSettings.thinking?.type === "enabled" && updatedSettings.thinking.budget_tokens) {
        updatedSettings.thinking = {
            ...updatedSettings.thinking,
            budget_tokens: Math.min(updatedSettings.thinking.budget_tokens, maxTokens - 1)
        };
    }

    return {
        ...updatedSettings,
        max_tokens: maxTokens
    };
}

/**
 * Validates and retrieves the max output tokens from environment or settings.
 */
export function getMaxOutputTokens(userSettings: any, envMaxTokens?: string): number {
    // In the original, this calls a validator. Here we provide a sane default and capped logic.
    let maxTokens = userSettings.maxOutputTokens || 4096;
    const envVal = parseInt(envMaxTokens || process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS || "4096", 10);

    if (!isNaN(envVal)) {
        return Math.min(envVal, maxTokens);
    }
    return maxTokens;
}

/**
 * Placeholder for message transformation for display.
 * Logic for bB2 and xB2 deobfuscation would be here.
 */
export function transformUserMessagesForDisplay(messages: Message[], cachingEnabled: boolean): Message[] {
    // Stub implementation
    return messages.map(msg => ({ ...msg }));
}
