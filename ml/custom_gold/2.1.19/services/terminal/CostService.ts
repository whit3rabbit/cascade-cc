/**
 * File: src/services/terminal/CostService.ts
 * Role: Tracks and estimates session cost based on token usage.
 */

export interface Usage {
    inputTokens: number;
    outputTokens: number;
    cacheWriteTokens?: number;
    cacheReadTokens?: number;
}

export interface ModelRates {
    input: number;      // per 1M tokens
    output: number;     // per 1M tokens
    cacheWrite?: number; // per 1M tokens
    cacheRead?: number;  // per 1M tokens
}

const RATES: Record<string, ModelRates> = {
    "claude-3-5-sonnet-20241022": {
        input: 3.00,
        output: 15.00,
        cacheWrite: 3.75,
        cacheRead: 0.30
    },
    "claude-3-opas-20240229": {
        input: 15.00,
        output: 75.00
    },
    "claude-3-haiku-20240307": {
        input: 0.25,
        output: 1.25
    },
    "claude-3-5-sonnet-20241022[1m]": {
        input: 3.00, // Same base input
        output: 15.00,
        cacheWrite: 3.75,
        cacheRead: 0.30
    }
};

class CostService {
    private totalUsage: Usage = {
        inputTokens: 0,
        outputTokens: 0,
        cacheWriteTokens: 0,
        cacheReadTokens: 0
    };

    /**
     * Records usage for a specific turn.
     */
    addUsage(usage: Usage): void {
        this.totalUsage.inputTokens += usage.inputTokens || 0;
        this.totalUsage.outputTokens += usage.outputTokens || 0;
        this.totalUsage.cacheWriteTokens = (this.totalUsage.cacheWriteTokens || 0) + (usage.cacheWriteTokens || 0);
        this.totalUsage.cacheReadTokens = (this.totalUsage.cacheReadTokens || 0) + (usage.cacheReadTokens || 0);
    }

    /**
     * Calculates the estimated cost in USD.
     */
    calculateCost(model: string = "claude-3-5-sonnet-20241022"): number {
        // Resolve extended context if present
        let lookupModel = model;
        // In reality, 1m might have different pricing.
        // For now, let's treat it as the base model or a specific key if defined.

        const defaultRates = RATES["claude-3-5-sonnet-20241022"]!;
        const rates = RATES[lookupModel] || RATES[model.replace(/\[1m\]$/, '')] || defaultRates;

        const inputCost = (this.totalUsage.inputTokens / 1_000_000) * rates.input;
        const outputCost = (this.totalUsage.outputTokens / 1_000_000) * rates.output;
        const cacheWriteCost = ((this.totalUsage.cacheWriteTokens || 0) / 1_000_000) * (rates.cacheWrite || rates.input);
        const cacheReadCost = ((this.totalUsage.cacheReadTokens || 0) / 1_000_000) * (rates.cacheRead || rates.input * 0.1);

        return inputCost + outputCost + cacheWriteCost + cacheReadCost;
    }

    /**
     * Returns the total usage.
     */
    getUsage(): Usage {
        return { ...this.totalUsage };
    }

    /**
     * Resets usage tracking.
     */
    reset(): void {
        this.totalUsage = {
            inputTokens: 0,
            outputTokens: 0,
            cacheWriteTokens: 0,
            cacheReadTokens: 0
        };
    }
}

export const costService = new CostService();
