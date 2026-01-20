/**
 * Feature usage tracking.
 * Deobfuscated from chunk_202.ts.
 */

import { featureTips } from "./featureTips.js";
import { getSettings, updateSettings } from "../terminal/settings.js";

/**
 * Increments usage count for a feature.
 * Deobfuscated from v9 in chunk_202.ts.
 */
export function trackFeatureUsage(featureId: string): void {
    const settings = getSettings("userSettings");
    const usage = settings.featureUsage || {};

    usage[featureId] = (usage[featureId] || 0) + 1;

    updateSettings("userSettings", { featureUsage: usage });
}

/**
 * Calculates how many features the user has explored.
 * Deobfuscated from UWB in chunk_202.ts.
 */
export async function getExplorationStats() {
    const tips = featureTips;
    const results = await Promise.all(tips.map(async (tip) => ({
        id: tip.id,
        categoryId: tip.categoryId,
        used: await tip.hasBeenUsed()
    })));

    let explored = 0;
    const byCategory: Record<string, { explored: number; total: number }> = {};

    for (const tip of tips) {
        if (!byCategory[tip.categoryId]) {
            byCategory[tip.categoryId] = { explored: 0, total: 0 };
        }
        byCategory[tip.categoryId].total++;
    }

    for (const res of results) {
        if (res.used) {
            explored++;
            byCategory[res.categoryId].explored++;
        }
    }

    return {
        explored,
        total: tips.length,
        byCategory
    };
}
