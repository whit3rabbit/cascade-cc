import { batchPromise } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { getGlobalState, setGlobalState } from "../session/globalState.js";
import { getSessionId, getOriginalCwd } from "../session/sessionStore.js";
import { log } from "../logger/loggerService.js";
import * as path from "path";
import * as fs from "fs";

const logger = log("claudeUtils");

// Bedrock command placeholders - should ideally be imported from @aws-sdk/client-bedrock
// but we'll use these simplified classes as they match the usage in the deobfuscated code.
import { BedrockClient, ListInferenceProfilesCommand, GetInferenceProfileCommand } from "@aws-sdk/client-bedrock";

/**
 * Gets a Bedrock client for the current region.
 */
let _bedrockClient: BedrockClient | null = null;
async function getBedrockClient(): Promise<BedrockClient> {
    if (_bedrockClient) return _bedrockClient;

    // Dynamically determine region - simplified for now
    const region = process.env.AWS_REGION || "us-east-1";
    _bedrockClient = new BedrockClient({ region });
    return _bedrockClient;
}

// Model IDs for different providers
export const MODELS_BY_PROVIDER = {
    firstParty: {
        haiku35: "claude-3-5-haiku-20241022",
        haiku45: "claude-haiku-4-5-20251001",
        sonnet35: "claude-3-5-sonnet-20241022",
        sonnet37: "claude-3-7-sonnet-20250219",
        sonnet40: "claude-sonnet-4-20250514",
        sonnet45: "claude-sonnet-4-5-20250929",
        opus40: "claude-opus-4-20250514",
        opus41: "claude-opus-4-1-20250805",
        opus45: "claude-opus-4-5-20251101"
    },
    bedrock: {
        haiku35: "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        haiku45: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        sonnet35: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        sonnet37: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        sonnet40: "us.anthropic.claude-sonnet-4-20250514-v1:0",
        sonnet45: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        opus40: "us.anthropic.claude-opus-4-20250514-v1:0",
        opus41: "us.anthropic.claude-opus-4-1-20250805-v1:0",
        opus45: "us.anthropic.claude-opus-4-5-20251101-v1:0"
    },
    vertex: {
        haiku35: "claude-3-5-haiku@20241022",
        haiku45: "claude-haiku-4-5@20251001",
        sonnet35: "claude-3-5-sonnet-v2@20241022",
        sonnet37: "claude-3-7-sonnet@20250219",
        sonnet40: "claude-sonnet-4@20250514",
        sonnet45: "claude-sonnet-4-5@20250929",
        opus40: "claude-opus-4@20250514",
        opus41: "claude-opus-4-1@20250805",
        opus45: "claude-opus-4-5@20251101"
    },
    foundry: {
        haiku35: "claude-3-5-haiku",
        haiku45: "claude-haiku-4-5",
        sonnet35: "claude-3-5-sonnet",
        sonnet37: "claude-3-7-sonnet",
        sonnet40: "claude-sonnet-4",
        sonnet45: "claude-sonnet-4-5",
        opus40: "claude-opus-4",
        opus41: "claude-opus-4-1",
        opus45: "claude-opus-4-5"
    }
};

export const MODEL_RATES_MAP = {
    haiku35: { inputTokens: 0.8, outputTokens: 4, promptCacheWriteTokens: 1, promptCacheReadTokens: 0.08, webSearchRequests: 0.01 },
    haiku45: { inputTokens: 1, outputTokens: 5, promptCacheWriteTokens: 1.25, promptCacheReadTokens: 0.1, webSearchRequests: 0.01 },
    sonnet: { inputTokens: 3, outputTokens: 15, promptCacheWriteTokens: 3.75, promptCacheReadTokens: 0.3, webSearchRequests: 0.01 },
    sonnetHighUsage: { inputTokens: 6, outputTokens: 22.5, promptCacheWriteTokens: 7.5, promptCacheReadTokens: 0.6, webSearchRequests: 0.01 },
    opus: { inputTokens: 15, outputTokens: 75, promptCacheWriteTokens: 18.75, promptCacheReadTokens: 1.5, webSearchRequests: 0.01 },
    opus45: { inputTokens: 5, outputTokens: 25, promptCacheWriteTokens: 6.25, promptCacheReadTokens: 0.5, webSearchRequests: 0.01 }
};

export const MODEL_CONTEXT_WINDOWS: Record<string, number> = {
    haiku: 200000,
    sonnet: 200000,
    opus: 200000,
    "sonnet[1m]": 1000000,
    "sonnet[2m]": 2000000
};

/**
 * Lists AWS Bedrock inference profiles for the current region.
 */
export const listInferenceProfiles = batchPromise(async function (): Promise<string[]> {
    try {
        const client = await getBedrockClient();
        const profiles: string[] = [];
        let nextToken: string | undefined;

        do {
            const command = new ListInferenceProfilesCommand({
                maxResults: 100,
                nextToken
            });
            const response: any = await client.send(command);
            if (response.inferenceProfileSummaries) {
                profiles.push(...response.inferenceProfileSummaries.map((p: any) => p.inferenceProfileArn));
            }
            nextToken = response.nextToken;
        } while (nextToken);

        return profiles;
    } catch (err) {
        logger.debug(`Failed to list inference profiles: ${err}`);
        return [];
    }
});

/**
 * Gets the base model ID for a given inference profile.
 */
export const getInferenceProfile = batchPromise(async function (profileId: string): Promise<string | null> {
    try {
        const client = await getBedrockClient();
        const command = new GetInferenceProfileCommand({
            inferenceProfileIdentifier: profileId
        });
        const response: any = await client.send(command);
        if (!response.models || response.models.length === 0) return null;

        const model = response.models[0];
        if (!model?.modelArn) return null;

        // Extract model ID from ARN
        const lastSlash = model.modelArn.lastIndexOf("/");
        return lastSlash >= 0 ? model.modelArn.substring(lastSlash + 1) : model.modelArn;
    } catch (err) {
        logger.error(`Failed to get inference profile ${profileId}: ${err}`);
        return null;
    }
});

/**
 * Resolves Bedrock model IDs to the best available inference profile or default ID.
 */
export async function resolveBedrockModelIds() {
    let profiles: string[];
    try {
        profiles = await listInferenceProfiles();
    } catch (err) {
        return MODELS_BY_PROVIDER.bedrock;
    }

    if (!profiles || profiles.length === 0) return MODELS_BY_PROVIDER.bedrock;

    const findProfile = (modelId: string) => profiles.find(p => p.includes(modelId));

    const result: any = {};
    for (const [key, defaultId] of Object.entries(MODELS_BY_PROVIDER.bedrock)) {
        // Try to find a matching profile for the model part of the ID
        const modelPart = defaultId.split(".").pop()?.split("-v1")[0];
        result[key] = (modelPart && findProfile(modelPart)) || defaultId;
    }
    return result;
}

let cachedModelIds: typeof MODELS_BY_PROVIDER.firstParty | null = null;
const PROVIDER: string = "firstParty"; // Placeholder, should be detected

export function initModelPricing() {
    if (cachedModelIds !== null) return;
    if (PROVIDER !== "bedrock") {
        cachedModelIds = MODELS_BY_PROVIDER.firstParty as any;
        return;
    }
    refreshModelPricing();
}

/**
 * Returns model pricing configuration and rates.
 * Logic from initModelPricing/getModelPricing in chunk_167.ts.
 */
export function getModelPricing() {
    return {
        ...MODELS_BY_PROVIDER.firstParty,
        rates: MODEL_RATES_MAP
    };
}

export const refreshModelPricing = batchPromise(async function () {
    if (cachedModelIds !== null) return;
    try {
        cachedModelIds = await resolveBedrockModelIds();
    } catch (err) {
        logger.error(`Failed to refresh model pricing: ${err}`);
    }
});

/**
 * Normalizes a model name by extracting the base Claude model identifier.
 */
export function normalizeModelName(name: string): string {
    if (!name) return name;
    if (name.includes("claude-opus-4-5")) return "claude-opus-4-5";
    if (name.includes("claude-opus-4-1")) return "claude-opus-4-1";
    if (name.includes("claude-opus-4")) return "claude-opus-4";

    const match = name.match(/(claude-(\d+-\d+-)?\w+)/);
    return (match && match[1]) ? match[1] : name;
}

/**
 * Returns the currently active model ID.
 * Logic from p3 in chunk_167.ts.
 */
export function getActiveModel(): string {
    if (process.env.CLAUDE_CODE_SUBAGENT_MODEL) return process.env.CLAUDE_CODE_SUBAGENT_MODEL;
    const name = getDefaultModelName();
    return resolveModelAlias(name);
}

/**
 * Gets the default model name, optionally formatted for display.
 * Logic from V1A/LrA in chunk_167.ts.
 */
export function getDefaultModelName(options: { forDisplay?: boolean } = {}): string {
    const { forDisplay = false } = options;

    const envModel = process.env.ANTHROPIC_MODEL;
    if (envModel) return forDisplay ? getModelDisplayName(envModel) : envModel;

    // Simplified fallback logic
    const defaultModel = MODELS_BY_PROVIDER.firstParty.sonnet37;
    return forDisplay ? getModelDisplayName(defaultModel) : defaultModel;
}

/**
 * Gets the display name for a given model.
 */
export function getModelDisplayName(modelName: string | null): string {
    if (!modelName) return "Claude";

    // Exhaustive mapping from tlQ in chunk_168
    const pricing = getModelPricing();
    switch (modelName) {
        case pricing.opus45: return "Opus 4.5";
        case pricing.opus41: return "Opus 4.1";
        case pricing.opus40: return "Opus 4";
        case pricing.sonnet45 + "[1m]": return "Sonnet 4.5 (1M context)";
        case pricing.sonnet45: return "Sonnet 4.5";
        case pricing.sonnet40 + "[1m]": return "Sonnet 4 (1M context)";
        case pricing.sonnet40: return "Sonnet 4";
        case pricing.sonnet37: return "Sonnet 3.7";
        case pricing.sonnet35: return "Sonnet 3.5";
        case pricing.haiku45: return "Haiku 4.5";
        case pricing.haiku35: return "Haiku 3.5";
    }

    // Fallback logic
    if (modelName.includes("claude-sonnet-4-5") && modelName.includes("[1m]")) return "Sonnet 4.5 (with 1M context)";
    if (modelName.includes("claude-sonnet-4-5")) return "Sonnet 4.5";
    if (modelName.includes("claude-sonnet-4") && modelName.includes("[1m]")) return "Sonnet 4 (with 1M context)";
    if (modelName.includes("claude-sonnet-4")) return "Sonnet 4";
    if (modelName.includes("claude-opus-4-5")) return "Opus 4.5";
    if (modelName.includes("claude-opus-4-1")) return "Opus 4.1";
    if (modelName.includes("claude-opus-4")) return "Opus 4";
    if (modelName.includes("claude-3-7-sonnet")) return "Claude 3.7 Sonnet";
    if (modelName.includes("claude-3-5-sonnet")) return "Claude 3.5 Sonnet";
    if (modelName.includes("claude-haiku-4-5")) return "Haiku 4.5";
    if (modelName.includes("claude-3-5-haiku")) return "Claude 3.5 Haiku";

    if (modelName === "sonnet") return "Sonnet";
    if (modelName === "opus") return "Opus";
    if (modelName === "haiku") return "Haiku";

    return (modelName.charAt(0).toUpperCase() + modelName.slice(1)).replace(/-/g, " ");
}

/**
 * Formats a model name with the "Claude" prefix if it's a known model.
 */
export function formatClaudeModelName(modelName: string): string {
    const displayName = getModelDisplayName(modelName);
    if (displayName && displayName !== "Claude" && !displayName.includes(" ")) {
        return `Claude ${displayName}`;
    }
    return displayName.startsWith("Claude") ? displayName : `Claude ${displayName}`;
}

/**
 * Resolves a model alias (e.g., 'sonnet', 'opus') to its full model ID.
 */
export function resolveModelAlias(alias: string): string {
    const trimmed = alias.trim();
    const lower = trimmed.toLowerCase();
    const has1M = lower.endsWith("[1m]");
    const base = has1M ? lower.replace(/\[1m]$/i, "").trim() : lower;

    if (isModelAlias(base)) {
        switch (base) {
            case "opusplan":
                return getDefaultSonnetModel() + (has1M ? "[1m]" : "");
            case "sonnet":
                return getDefaultSonnetModel() + (has1M ? "[1m]" : "");
            case "haiku":
                return getDefaultHaikuModel() + (has1M ? "[1m]" : "");
            case "opus":
                return getDefaultOpusModel();
            default:
        }
    }

    if (has1M) return trimmed.replace(/\[1m\]$/i, "").trim() + "[1m]";
    return trimmed;
}

export function isModelAlias(name: string): boolean {
    return ["sonnet", "opus", "haiku", "sonnet[1m]", "opusplan"].includes(name);
}

export function getDefaultSonnetModel(): string {
    if (process.env.ANTHROPIC_DEFAULT_SONNET_MODEL) return process.env.ANTHROPIC_DEFAULT_SONNET_MODEL;
    return MODELS_BY_PROVIDER.firstParty.sonnet45;
}

export function getDefaultOpusModel(): string {
    if (process.env.ANTHROPIC_DEFAULT_OPUS_MODEL) return process.env.ANTHROPIC_DEFAULT_OPUS_MODEL;
    // In actual code, provider check is done: x4() === "firstParty"
    return MODELS_BY_PROVIDER.firstParty.opus45;
}

export function getDefaultHaikuModel(): string {
    if (process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL) return process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL;
    // In actual code: x4() === "firstParty" || x4() === "foundry"
    return MODELS_BY_PROVIDER.firstParty.haiku45;
}

/**
 * Calculates the cost of a single model request.
 */
export function calculateUsageCost(modelId: string, usage: any): number {
    const rates = getRatesForModel(modelId, usage);
    return calculateUsageCostInternal(rates, usage);
}

function calculateUsageCostInternal(rates: any, usage: any): number {
    if (!usage) return 0;

    const inputCost = (usage.input_tokens / 1000000) * rates.inputTokens;
    const outputCost = (usage.output_tokens / 1000000) * rates.outputTokens;
    const cacheReadCost = ((usage.cache_read_input_tokens || 0) / 1000000) * rates.promptCacheReadTokens;
    const cacheWriteCost = ((usage.cache_creation_input_tokens || 0) / 1000000) * rates.promptCacheWriteTokens;
    const webSearchCost = (usage.server_tool_use?.web_search_requests || 0) * (rates.webSearchRequests || 0);

    return inputCost + outputCost + cacheReadCost + cacheWriteCost + webSearchCost;
}

function getRatesForModel(modelId: string, usage: any): any {
    const baseName = normalizeModelName(modelId);

    // Logic from MD8
    if (baseName.includes("sonnet")) {
        const totalTokens = (usage.input_tokens || 0) + (usage.cache_read_input_tokens || 0) + (usage.cache_creation_input_tokens || 0);
        if (totalTokens > 200000) return MODEL_RATES_MAP.sonnetHighUsage;
        return MODEL_RATES_MAP.sonnet;
    }

    if (baseName.includes("opus-4-5")) return MODEL_RATES_MAP.opus45;
    if (baseName.includes("opus")) return MODEL_RATES_MAP.opus;
    if (baseName.includes("haiku-4-5")) return MODEL_RATES_MAP.haiku45;
    if (baseName.includes("haiku")) return MODEL_RATES_MAP.haiku35;

    return MODEL_RATES_MAP.sonnet; // Default
}

/**
 * Persists current session usage stats to disk.
 */
export function persistUsageStats() {
    try {
        const state = getGlobalState();
        // logic from Og1
        setGlobalState({
            lastCost: state.totalCostUSD,
            lastAPIDuration: state.totalAPIDuration,
            // ... more stats
            lastSessionId: getSessionId()
        } as any);

        const statsDir = path.join(process.cwd(), ".claude", "stats");
        if (!fs.existsSync(statsDir)) {
            fs.mkdirSync(statsDir, { recursive: true });
        }

        const statsFile = path.join(statsDir, `${getSessionId()}.json`);
        const stats = {
            totalCostUSD: state.totalCostUSD,
            modelUsage: state.modelUsage,
            startTime: state.startTime,
            totalAPIDuration: state.totalAPIDuration
        };

        fs.writeFileSync(statsFile, JSON.stringify(stats, null, 2), "utf8");
    } catch (err) {
        logger.debug(`Failed to persist usage stats: ${err}`);
    }
}


/**
 * Formats a duration in milliseconds to a human-readable string.
 */
export function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) {
        const seconds = (ms / 1000).toFixed(1);
        return `${seconds}s`;
    }

    const days = Math.floor(ms / 86400000);
    const hours = Math.floor((ms % 86400000) / 3600000);
    const minutes = Math.floor((ms % 3600000) / 60000);
    const seconds = Math.round((ms % 60000) / 1000);

    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
}

/**
 * Formats a number for compact display (e.g., 1.2k, 3.4M).
 */
export function formatNumberCompact(num: number): string {
    const isLarge = num >= 1000;
    return new Intl.NumberFormat("en", {
        notation: "compact",
        minimumFractionDigits: isLarge ? 1 : 0,
        maximumFractionDigits: 1
    }).format(num).toLowerCase();
}

export function formatCost(amount: number, fractionDigits: number = 4): string {
    if (amount > 0.5) return `$${amount.toFixed(2)}`;
    return `$${amount.toFixed(fractionDigits)}`;
}

/**
 * Formats a date into a relative time string (e.g., "2h ago").
 */
export function formatRelativeTime(date: Date, options: {
    style?: "narrow" | "short" | "long",
    numeric?: "always" | "auto",
    now?: Date
} = {}): string {
    const {
        style = "narrow",
        numeric = "always",
        now = new Date()
    } = options;

    const diffMs = date.getTime() - now.getTime();
    const diffSec = Math.trunc(diffMs / 1000);

    const units: { unit: Intl.RelativeTimeFormatUnit; seconds: number; shortUnit: string }[] = [
        { unit: "year", seconds: 31536000, shortUnit: "y" },
        { unit: "month", seconds: 2592000, shortUnit: "mo" },
        { unit: "week", seconds: 604800, shortUnit: "w" },
        { unit: "day", seconds: 86400, shortUnit: "d" },
        { unit: "hour", seconds: 3600, shortUnit: "h" },
        { unit: "minute", seconds: 60, shortUnit: "m" },
        { unit: "second", seconds: 1, shortUnit: "s" }
    ];

    for (const { unit, seconds, shortUnit } of units) {
        if (Math.abs(diffSec) >= seconds) {
            const val = Math.trunc(diffSec / seconds);
            if (style === "narrow") {
                return diffSec < 0 ? `${Math.abs(val)}${shortUnit} ago` : `in ${val}${shortUnit}`;
            }
            return new Intl.RelativeTimeFormat("en", {
                style: "long",
                numeric
            }).format(val, unit);
        }
    }

    if (style === "narrow") return diffSec <= 0 ? "0s ago" : "in 0s";
    return new Intl.RelativeTimeFormat("en", {
        style,
        numeric
    }).format(0, "second");
}

/**
 * Always formats as relative time, ensuring "always" numeric style for past dates.
 */
export function formatRelativeTimeAlways(date: Date, options: { now?: Date } = {}): string {
    const { now = new Date() } = options;
    if (date > now) return formatRelativeTime(date, { now });
    return formatRelativeTime(date, { numeric: "always", now });
}

/**
 * Formats a Unix timestamp.
 */
export function formatTimestamp(timestamp: number | undefined, options: { showTimeZone?: boolean, showTime?: boolean } = {}): string | undefined {
    if (!timestamp) return undefined;

    const date = new Date(timestamp * 1000);
    const now = new Date();
    const { showTimeZone = false, showTime = true } = options;
    const minutes = date.getMinutes();

    // If more than 24 hours diff
    if (Math.abs(date.getTime() - now.getTime()) > 86400000) {
        const fmtOptions: Intl.DateTimeFormatOptions = {
            month: "short",
            day: "numeric",
            hour: showTime ? "numeric" : undefined,
            minute: !showTime || minutes === 0 ? undefined : "2-digit",
            hour12: showTime ? true : undefined
        };
        if (date.getFullYear() !== now.getFullYear()) {
            fmtOptions.year = "numeric";
        }
        let result = date.toLocaleString("en-US", fmtOptions).replace(/ ([AP]M)/i, (_, p1) => p1.toLowerCase());
        if (showTimeZone) {
            result += ` (${Intl.DateTimeFormat().resolvedOptions().timeZone})`;
        }
        return result;
    }

    const timeStr = date.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: minutes === 0 ? undefined : "2-digit",
        hour12: true
    }).replace(/ ([AP]M)/i, (_, p1) => p1.toLowerCase());

    const tzStr = showTimeZone ? ` (${Intl.DateTimeFormat().resolvedOptions().timeZone})` : "";
    return timeStr + tzStr;
}

export function formatDate(date: string | number | Date, showTimeZone: boolean = false, showTime: boolean = true): string {
    const d = new Date(date);
    const ts = Math.floor(d.getTime() / 1000);
    return formatTimestamp(ts, { showTimeZone, showTime }) || "";
}

export function truncateString(str: string, length: number, smart: boolean = false): string {
    let result = str;
    if (smart) {
        const newlineIndex = str.indexOf("\n");
        if (newlineIndex !== -1) {
            result = str.substring(0, newlineIndex);
            if (result.length + 1 > length) return `${result.substring(0, length - 1)}…`;
            return `${result}…`;
        }
    }
    if (result.length <= length) return result;
    return `${result.substring(0, length - 1)}…`;
}
