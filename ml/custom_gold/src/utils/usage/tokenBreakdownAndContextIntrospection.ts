import {
    formatDuration,
    formatCompactNumber,
} from "../shared/lodashLikeRuntimeAndEnv.js";
import { calculateModelUsageCost, formatUSD } from "./contextAndCostPricing.js";

// External state/metrics placeholders - these would typically come from a global state manager
declare function getGlobalState(): any;
declare function setGlobalState(fn: (state: any) => any): void;
declare function getSessionId(): string;
declare function getTotalCost(): number;
declare function getTotalApiDuration(): number;
declare function getTotalApiDurationWithoutRetries(): number;
declare function getTotalToolDuration(): number;
declare function getWallDuration(): number;
declare function getLinesAdded(): number;
declare function getLinesRemoved(): number;
declare function getModelUsageStats(): Record<string, any>;

/**
 * Aggregates stats for the current session.
 */
export function getSessionStats(sessionId: string) {
    const state = getGlobalState();
    if (state.lastSessionId !== sessionId) return;

    let modelUsage: any;
    if (state.lastModelUsage) {
        modelUsage = Object.fromEntries(
            Object.entries(state.lastModelUsage).map(([model, stats]: [string, any]) => [
                model,
                { ...stats, contextWindow: 0 } // contextWindow lookup would go here
            ])
        );
    }

    return {
        totalCostUSD: state.lastCost ?? 0,
        totalAPIDuration: state.lastAPIDuration ?? 0,
        totalAPIDurationWithoutRetries: state.lastAPIDurationWithoutRetries ?? 0,
        totalToolDuration: state.lastToolDuration ?? 0,
        totalLinesAdded: state.lastLinesAdded ?? 0,
        totalLinesRemoved: state.lastLinesRemoved ?? 0,
        lastDuration: state.lastDuration,
        modelUsage
    };
}

/**
 * Persists current session usage stats to the global state.
 */
export function recordSessionUsage() {
    setGlobalState((state: any) => ({
        ...state,
        lastCost: getTotalCost(),
        lastAPIDuration: getTotalApiDuration(),
        lastAPIDurationWithoutRetries: getTotalApiDurationWithoutRetries(),
        lastToolDuration: getTotalToolDuration(),
        lastDuration: getWallDuration(),
        lastLinesAdded: getLinesAdded(),
        lastLinesRemoved: getLinesRemoved(),
        lastModelUsage: Object.fromEntries(
            Object.entries(getModelUsageStats()).map(([model, stats]) => [
                model,
                {
                    inputTokens: stats.inputTokens,
                    outputTokens: stats.outputTokens,
                    cacheReadInputTokens: stats.cacheReadInputTokens,
                    cacheCreationInputTokens: stats.cacheCreationInputTokens,
                    webSearchRequests: stats.webSearchRequests,
                    costUSD: stats.costUSD
                }
            ])
        ),
        lastSessionId: getSessionId()
    }));
}

/**
 * Formats token usage by model into a string.
 */
export function formatModelUsage(): string {
    const usage = getModelUsageStats();
    if (Object.keys(usage).length === 0) {
        return "Usage:                 0 input, 0 output, 0 cache read, 0 cache write";
    }

    let result = "Usage by model:";
    for (const [model, stats] of Object.entries(usage)) {
        const usageStr = `  ${formatCompactNumber(stats.inputTokens)} input, ${formatCompactNumber(stats.outputTokens)} output, ${formatCompactNumber(stats.cacheReadInputTokens)} cache read, ${formatCompactNumber(stats.cacheCreationInputTokens)} cache write` +
            (stats.webSearchRequests > 0 ? `, ${formatCompactNumber(stats.webSearchRequests)} web search` : "") +
            ` (${formatUSD(stats.costUSD)})`;

        result += `\n${(model + ":").padStart(21)}${usageStr}`;
    }
    return result;
}

/**
 * Formats total session usage summary.
 */
export function formatTotalUsage(): string {
    const totalCost = formatUSD(getTotalCost());
    const usageStr = formatModelUsage();

    return `Total cost:            ${totalCost}
Total duration (API):  ${formatDuration(getTotalApiDuration())}
Total duration (wall): ${formatDuration(getWallDuration())}
Total code changes:    ${getLinesAdded()} ${getLinesAdded() === 1 ? "line" : "lines"} added, ${getLinesRemoved()} ${getLinesRemoved() === 1 ? "line" : "lines"} removed
${usageStr}`;
}

/**
 * Registers a process exit hook to log final usage stats.
 */
export function registerUsageExitHandler() {
    const handler = () => {
        // In a real CLI, we might check if we should print based on a flag
        console.log(formatTotalUsage());
        recordSessionUsage();
    };
    process.on("exit", handler);
    return () => process.off("exit", handler);
}

/**
 * Records usage for a specific model call.
 */
export function recordModelUsage(model: string, usage: any) {
    const cost = calculateModelUsageCost(model, usage);
    // This would update internal counters and metrics
    // recordMetrics(model, usage, cost);
}
