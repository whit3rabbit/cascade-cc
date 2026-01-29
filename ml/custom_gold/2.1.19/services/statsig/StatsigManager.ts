/**
 * File: src/services/statsig/StatsigManager.ts
 * Role: Higher-level manager for Statsig feature gates and experiments.
 */

import { createHash } from "node:crypto";
import { StatsigClient, StatsigUser } from "./StatsigClientBase.js";

let clientInstance: StatsigClient | null = null;
let initializationPromise: Promise<void> | null = null;

const REFRESH_INTERVAL_MS = 6 * 60 * 60 * 1000; // 6 hours

/**
 * Initializes the Statsig feature flag client.
 * 
 * @param sdkKey - The Statsig client SDK key.
 * @param userData - User metadata for targeting.
 */
export async function initializeStatsig(sdkKey: string, userData: StatsigUser = { userId: "" }): Promise<void> {
    if (initializationPromise) return initializationPromise;

    const options = {
        networkOptions: {
            apiEndpoint: "https://statsig.anthropic.com/v1/",
            networkTimeoutMilliseconds: 30000,
        },
        statsigEnvironment: { deploymentTier: "production" },
        customUserCacheKey: (w: string, user: StatsigUser) => {
            return createHash("sha1")
                .update(w)
                .update(user.userId || "")
                .digest("hex")
                .slice(0, 10);
        }
    };

    const client = new StatsigClient(sdkKey, userData, options);
    clientInstance = client;
    initializationPromise = client.initializeAsync();

    // Setup periodic refresh
    setInterval(() => {
        if (clientInstance) {
            clientInstance.updateUserAsync(userData).catch(console.error);
        }
    }, REFRESH_INTERVAL_MS);

    return initializationPromise;
}

/**
 * Checks if a specific feature gate is enabled for the current user.
 * 
 * @param gateName - The name of the feature gate.
 * @returns True if the gate is enabled.
 */
export function checkGate(gateName: string): boolean {
    if (!clientInstance) return false;
    return clientInstance.checkGate(gateName);
}

/**
 * Gets a value from a Statsig experiment.
 */
export function getExperimentValue(experimentName: string, key: string, defaultValue: any): any {
    if (!clientInstance) return defaultValue;
    const experiment = clientInstance.getExperiment(experimentName);
    return experiment?.get(key, defaultValue) ?? defaultValue;
}
