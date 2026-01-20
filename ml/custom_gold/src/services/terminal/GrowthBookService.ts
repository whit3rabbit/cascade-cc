
/**
 * Service for feature flags and experiments using GrowthBook logic.
 * Logic from chunk_589.ts, chunk_61.ts
 */

import { mergeSettings, getSettings, updateSettings } from "./settings.js";
import { log } from "../logger/loggerService.js";

const logger = log("growthbook");

let cachedFeatures: Record<string, any> = {};
let cachedGates: Record<string, boolean> = {};

/**
 * Checks if a feature gate is enabled.
 * Deobfuscated from gZ in chunk_589.ts:265
 */
export function checkStatsigGate(gateName: string): boolean {
    const settings = mergeSettings();

    // Check cached growthbook features first
    if (settings.cachedGrowthBookFeatures && gateName in settings.cachedGrowthBookFeatures) {
        return Boolean(settings.cachedGrowthBookFeatures[gateName]);
    }

    // Fallback to cached statsig gates
    if (settings.cachedStatsigGates && gateName in settings.cachedStatsigGates) {
        return Boolean(settings.cachedStatsigGates[gateName]);
    }

    return false;
}

/**
 * Gets a feature flag value.
 * Deobfuscated from $M in chunk_589.ts:253
 */
export function getFeatureValue<T>(featureName: string, defaultValue: T): T {
    const settings = mergeSettings();

    if (settings.cachedGrowthBookFeatures && featureName in settings.cachedGrowthBookFeatures) {
        return settings.cachedGrowthBookFeatures[featureName] as T;
    }

    return defaultValue;
}

/**
 * Forces a refresh of GrowthBook features.
 * Simplified bridge for chunk_589:275 ($M2)
 */
export async function refreshGrowthBook() {
    // In a real implementation, this would call the GrowthBook API
    // and then update the cached settings.
    logger.debug("GrowthBook refresh requested (stub)");
}

/**
 * Records feature usage for analytics/telemetry.
 */
export function recordFeatureUsage(featureName: string) {
    // Logic from chunk_589:190 (ND1)
    // In original code this triggers a telemetry event
    logger.debug(`Feature used: ${featureName}`);
}
