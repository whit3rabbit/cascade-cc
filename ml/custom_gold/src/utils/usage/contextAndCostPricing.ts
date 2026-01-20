export interface ModelRates {
    inputTokens: number;
    outputTokens: number;
    promptCacheWriteTokens: number;
    promptCacheReadTokens: number;
    webSearchRequests: number;
}

export const PRICING_SONNET_3_5: ModelRates = {
    inputTokens: 3,
    outputTokens: 15,
    promptCacheWriteTokens: 3.75,
    promptCacheReadTokens: 0.3,
    webSearchRequests: 0.01
};

export const PRICING_OPUS: ModelRates = {
    inputTokens: 15,
    outputTokens: 75,
    promptCacheWriteTokens: 18.75,
    promptCacheReadTokens: 1.5,
    webSearchRequests: 0.01
};

export const PRICING_OPUS_4_5: ModelRates = {
    inputTokens: 5,
    outputTokens: 25,
    promptCacheWriteTokens: 6.25,
    promptCacheReadTokens: 0.5,
    webSearchRequests: 0.01
};

export const PRICING_HAIKU: ModelRates = {
    inputTokens: 6,
    outputTokens: 22.5,
    promptCacheWriteTokens: 7.5,
    promptCacheReadTokens: 0.6,
    webSearchRequests: 0.01
};

export const PRICING_HAIKU_3: ModelRates = {
    inputTokens: 0.8,
    outputTokens: 4,
    promptCacheWriteTokens: 1,
    promptCacheReadTokens: 0.08,
    webSearchRequests: 0.01
};

export const PRICING_HAIKU_3_5: ModelRates = {
    inputTokens: 1,
    outputTokens: 5,
    promptCacheWriteTokens: 1.25,
    promptCacheReadTokens: 0.1,
    webSearchRequests: 0.01
};

// Simplified model mapping for now
export const MODEL_PRICING_MAP: Record<string, ModelRates> = {
    "claude-3-5-sonnet": PRICING_SONNET_3_5,
    "claude-3-7-sonnet": PRICING_SONNET_3_5,
    "claude-3-opus": PRICING_OPUS,
    "claude-3-5-haiku": PRICING_HAIKU_3_5,
    "claude-3-haiku": PRICING_HAIKU_3,
};

/**
 * Calculates the cost for a given set of tokens using the provided rates.
 */
export function calculateTokensCost(rates: ModelRates, usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
    server_tool_use?: { web_search_requests?: number };
}): number {
    return (
        (usage.input_tokens / 1_000_000) * rates.inputTokens +
        (usage.output_tokens / 1_000_000) * rates.outputTokens +
        ((usage.cache_read_input_tokens ?? 0) / 1_000_000) * rates.promptCacheReadTokens +
        ((usage.cache_creation_input_tokens ?? 0) / 1_000_000) * rates.promptCacheWriteTokens +
        ((usage.server_tool_use?.web_search_requests ?? 0) * rates.webSearchRequests)
    );
}

/**
 * Calculates total tokens (input + cache).
 */
export function calculateTotalTokens(usage: {
    input_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
}): number {
    return usage.input_tokens + (usage.cache_read_input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0);
}

/**
 * Gets the rate info for a specific model.
 */
export function getModelRateInfo(modelName: string): ModelRates {
    // Logic from MD8: tries to resolve short name, defaults to a fallback if unknown
    const rates = MODEL_PRICING_MAP[modelName] || MODEL_PRICING_MAP["claude-3-5-sonnet"];
    return rates;
}

/**
 * Calculates the model usage cost in USD.
 */
export function calculateModelUsageCost(modelName: string, usage: any): number {
    const rates = getModelRateInfo(modelName);
    return calculateTokensCost(rates, usage);
}

/**
 * Formats a value as USD.
 */
export function formatUSD(value: number): string {
    if (Number.isInteger(value)) return `$${value}`;
    return `$${value.toFixed(2)}`;
}

/**
 * Formats model rates (e.g., $3/$15 per Mtok).
 */
export function formatModelRates(rates: ModelRates): string {
    return `${formatUSD(rates.inputTokens)}/${formatUSD(rates.outputTokens)} per Mtok`;
}
