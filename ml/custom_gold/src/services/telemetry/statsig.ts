import { createHash } from "node:crypto";
import { getTelemetryContext, prepareTelemetryEvent, encodeTelemetryMetadata } from "../../utils/shared/envContext.js";

// Statsig types and client placeholder
export enum StatsigLogLevel {
    None = 0,
    Info = 1,
    Debug = 2
}

export interface StatsigUser {
    userID?: string;
    custom?: Record<string, string | number | boolean>;
    [key: string]: any;
}

// Internal mocked client because we don't have the actual library installed in this repo environment
// In the real app, this would be from @statsig/js-client or similar
let statsigClientInstance: any = null;
let isInitialized = false;

/**
 * Initializes the Statsig client.
 */
export function initStatsigClient(sdkKey: string, user: StatsigUser) {
    const options = {
        networkConfig: {
            api: "https://statsig.anthropic.com/v1/"
        },
        environment: {
            tier: process.env.NODE_ENV === "production" ? "production" : "development"
        },
        includeCurrentPageUrlWithEvents: false,
        logLevel: StatsigLogLevel.None,
        customUserCacheKeyFunc: (user: StatsigUser, context: any) => {
            return createHash("sha1")
                .update(JSON.stringify(user))
                .update(context.userID || "")
                .digest("hex")
                .slice(0, 10);
        }
    };

    // Mocked client initialization logic
    statsigClientInstance = {
        initializeAsync: async () => { isInitialized = true; },
        on: (event: string, cb: Function) => { },
        logEvent: (event: any) => { },
        flush: async () => { },
        updateUserAsync: async (user: StatsigUser) => { },
        getDynamicConfig: (name: string) => ({ value: {} }),
        getExperiment: (name: string) => ({ get: (key: string, def: any) => def }),
        checkGate: (name: string) => false
    };

    const initPromise = statsigClientInstance.initializeAsync();

    process.on("beforeExit", async () => {
        if (statsigClientInstance) await statsigClientInstance.flush();
    });

    process.on("exit", () => {
        if (statsigClientInstance) statsigClientInstance.flush();
    });

    return {
        client: statsigClientInstance,
        initialized: initPromise
    };
}

/**
 * Internal helper for logging Statsig events.
 */
async function logStatsigEventInternal(eventName: string, options: { model?: string } = {}) {
    if (!isInitialized || !statsigClientInstance) return;

    try {
        const context = await getTelemetryContext({ model: options.model });
        const metadata = encodeTelemetryMetadata(context, options);
        const event = {
            eventName,
            metadata
        };
        statsigClientInstance.logEvent(event);
        await statsigClientInstance.flush();
    } catch (err) {
        // Ignore telemetry errors
    }
}

/**
 * Public helper for logging Statsig events.
 */
export function logStatsigEvent(eventName: string, options: any) {
    logStatsigEventInternal(eventName, options);
}

/**
 * Gets value from a Statsig dynamic config.
 */
export async function getStatsigDynamicConfig(configName: string, defaultValue: any) {
    if (!isInitialized || !statsigClientInstance) return defaultValue;
    const config = statsigClientInstance.getDynamicConfig(configName);
    if (Object.keys(config.value).length === 0) return defaultValue;
    return config.value;
}

/**
 * Gets value from a Statsig experiment.
 */
export function getStatsigExperimentValue(experimentName: string, key: string, defaultValue: any) {
    if (!isInitialized || !statsigClientInstance) return defaultValue;
    const experiment = statsigClientInstance.getExperiment(experimentName);
    return experiment.get(key, defaultValue);
}

/**
 * Checks a Statsig feature gate.
 */
export async function checkStatsigGate(gateName: string): Promise<boolean> {
    if (!isInitialized || !statsigClientInstance) return false;
    return statsigClientInstance.checkGate(gateName);
}

/**
 * Returns a cached Statsig gate value (stub).
 */
export function getCachedStatsigGate(gateName: string): boolean {
    // Logic from Wg: would check a local cache in global state
    return false;
}
