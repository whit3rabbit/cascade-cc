import { getActiveModel, resolveModelAlias, getModelDisplayName, formatCost, getModelPricing, MODEL_RATES_MAP, formatClaudeModelName } from "./claudeUtils.js";
import { isOauthActive } from "../auth/oauthManager.js";
import { getGlobalState } from "../session/globalState.js";

const BUILT_IN_MODELS = ["sonnet", "opus", "haiku", "sonnet[1m]", "opusplan"];

export interface ModelOption {
    value: string | null;
    label: string;
    description: string;
    descriptionForModel?: string;
}

/**
 * Strips context window suffixes from model names.
 */
export function stripContextWindow(modelName: string): string {
    return modelName.replace(/\[(1|2)m\]/gi, "");
}

/**
 * Normalizes model names, resolving aliases to actual IDs.
 * Logic from $K in chunk_168.ts.
 */
export function normalizeModelId(modelId: string): string {
    return resolveModelAlias(modelId);
}

/**
 * Resolves a model ID to its human-readable display name.
 * Delegated to getModelDisplayName in claudeUtils.ts.
 */
export function resolveModelDisplayName(modelId: string): string | null {
    const displayName = getModelDisplayName(modelId);
    return displayName === "Claude" ? null : displayName;
}

/**
 * Formats a model name for display.
 */
export function formatModelName(modelName: string): string {
    const resolved = getModelDisplayName(modelName);
    return resolved || modelName;
}

/**
 * Returns a human-readable label for a model.
 */
export function getModelLabel(modelId: string): string {
    if (modelId === "opusplan") return "Opus Plan";
    if (BUILT_IN_MODELS.includes(modelId)) {
        return modelId.charAt(0).toUpperCase() + modelId.slice(1);
    }
    return formatModelName(modelId);
}

/**
 * Returns a long description for a model.
 */
export function getModelDescriptionLong(modelId: string): string {
    if (modelId === "opusplan") return "Opus 4.5 in plan mode, else Sonnet 4.5";
    return formatModelName(normalizeModelId(modelId));
}

/**
 * Returns the description of the current user plan.
 * Logic from zqA in chunk_168.ts.
 */
export function getPlanDescription(): string {
    // These checks correspond to feature flags and subscription status
    // Simplified for now based on chunk_168:32-37
    return "Sonnet 4.5 · Best for everyday tasks";
}

/**
 * Formats the model rates for display.
 */
function formatModelRates(rates: { inputTokens: number; outputTokens: number }): string {
    return `$${rates.inputTokens}/$${rates.outputTokens} per Mtok`;
}

/**
 * Returns the default model option for selection.
 */
export function getDefaultModelOption(): ModelOption {
    const currentModel = getActiveModel();

    // Logic from Xi in chunk_168.ts
    if (isOauthActive()) {
        // ... (subscription checks)
        return {
            value: null,
            label: "Default (recommended)",
            description: `Default (${getPlanDescription()})`
        };
    }

    return {
        value: null,
        label: "Default (recommended)",
        description: `Use the default model (currently ${getModelLabel(currentModel)}) · Balanced`
    };
}

/**
 * Returns the available model options for the user.
 * Logic from mD8/GiQ in chunk_168.ts.
 */
export function getModelOptions(): ModelOption[] {
    const rates = MODEL_RATES_MAP;

    return [
        getDefaultModelOption(),
        {
            value: "sonnet",
            label: "Sonnet",
            description: `Sonnet 4.5 · Best for everyday tasks · ${formatModelRates(rates.sonnet)}`,
            descriptionForModel: "Sonnet 4.5 - best for everyday tasks. Generally recommended for most coding tasks"
        },
        {
            value: "sonnet[1m]",
            label: "Sonnet (1M context)",
            description: `Sonnet 4.5 for long sessions · ${formatModelRates(rates.sonnetHighUsage)}`,
            descriptionForModel: "Sonnet 4.5 with 1M context window - for long sessions with large codebases"
        },
        {
            value: "opus",
            label: "Opus",
            description: `Opus 4.5 · Most capable for complex work · ${formatModelRates(rates.opus45)}`,
            descriptionForModel: "Opus 4.5 - most capable for complex work. Generally more expensive than Sonnet."
        },
        {
            value: "haiku",
            label: "Haiku",
            description: `Haiku 4.5 · Fastest for quick answers · ${formatModelRates(rates.haiku45)}`,
            descriptionForModel: "Haiku 4.5 - fastest for quick answers. Lower cost but less capable than Sonnet 4.5."
        }
    ];
}

/**
 * Gets subagent model based on various settings.
 * Logic from qZA in chunk_168.ts.
 */
export function getSubagentModel(
    configModel?: string | null,
    mainLoopModel?: string,
    attachmentModel?: string | null,
    permissionMode: string = "default"
): string {
    if (process.env.CLAUDE_CODE_SUBAGENT_MODEL) return process.env.CLAUDE_CODE_SUBAGENT_MODEL;
    if (attachmentModel) return normalizeModelId(attachmentModel);

    const configured = configModel ?? process.env.ANTHROPIC_SUBAGENT_MODEL;
    if (!configured) return normalizeModelId("sonnet"); // Tg1 default

    if (configured === "inherit") {
        return normalizeModelId(mainLoopModel ?? "sonnet");
    }

    return normalizeModelId(configured);
}

/**
 * Formats subagent model name for display.
 * Logic from MrA in chunk_168.ts.
 */
export function formatSubagentModelName(modelId?: string | null): string {
    if (!modelId) return "Sonnet (default)";
    if (modelId === "inherit") return "Inherit from parent";
    return modelId.charAt(0).toUpperCase() + modelId.slice(1);
}
