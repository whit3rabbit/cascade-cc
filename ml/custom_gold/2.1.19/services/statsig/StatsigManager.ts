/**
 * File: src/services/statsig/StatsigManager.ts
 * Role: Higher-level manager for Statsig feature gates and experiments.
 */

import { StatsigClientBase, StatsigUser, StatsigClientOptions } from "./StatsigClientBase.js";
import { Log, Diagnostics } from "./StatsigService.js";

export class StatsigClient extends StatsigClientBase {
    constructor(sdkKey: string, user: StatsigUser, options: StatsigClientOptions = {}) {
        super(sdkKey, user, options);
    }

    async initializeAsync(options?: any): Promise<any> {
        if (this._initializePromise) {
            return this._initializePromise;
        }
        this._initializePromise = this._initializeAsyncImpl(options);
        return this._initializePromise;
    }

    private async _initializeAsyncImpl(options?: any): Promise<any> {
        Log.info("Starting Statsig initialization...");
        this._setStatus("Initializing", null);
        return this.updateUserAsync(this._user, options);
    }

    async updateUserAsync(newUser: StatsigUser, options?: any): Promise<any> {
        this._user = newUser;
        Diagnostics.markInitOverallStart(this._sdkKey);
        this._setStatus("Loading", null);

        try {
            // In a real implementation, this would perform a network request to fetch latest gates
            // For this deobfuscated version, we simulate the network request and process
            Diagnostics.markInitNetworkReqStart(this._sdkKey, { url: "/initialize" });

            // Mock delay
            await new Promise(resolve => setTimeout(resolve, 500));

            Diagnostics.markInitNetworkReqEnd(this._sdkKey, { success: true });
            Diagnostics.markInitProcessStart(this._sdkKey);

            // Mock values updated
            const mockValues = { source: "Network" };
            this._setStatus("Ready", mockValues);

            Diagnostics.markInitProcessEnd(this._sdkKey, true);
            Diagnostics.markInitOverallEnd(this._sdkKey, true, { source: "Network" });

            Log.info("Statsig updated successfully.");
            return true;
        } catch (error) {
            Diagnostics.markInitOverallEnd(this._sdkKey, false, { error });
            this._setStatus("error", null);
            Log.error("Statsig update failed:", error);
            throw error;
        }
    }

    checkGate(gateName: string): boolean {
        // Implementation based on _getFeatureGateImpl in chunk 414
        if (this.loadingStatus !== "Ready") return false;

        return this._memoize("gate", (name) => {
            // Mock logic: allow all gates for now as in original deobfuscation
            return true;
        })(gateName);
    }

    getExperiment(experimentName: string) {
        return {
            get: (key: string, defaultValue: any) => {
                if (this.loadingStatus !== "Ready") return defaultValue;
                return defaultValue;
            }
        };
    }
}

let clientInstance: StatsigClient | null = null;
let initializationPromise: Promise<void> | null = null;

const REFRESH_INTERVAL_MS = 6 * 60 * 60 * 1000; // 6 hours

export async function initializeStatsig(sdkKey: string, userData: StatsigUser = { userId: "" }): Promise<void> {
    if (initializationPromise) return initializationPromise;

    const options: StatsigClientOptions = {
        apiEndpoint: "https://statsig.anthropic.com/v1/",
        statsigEnvironment: { deploymentTier: "production" }
    };

    clientInstance = new StatsigClient(sdkKey, userData, options);
    initializationPromise = clientInstance.initializeAsync();

    // Setup periodic refresh
    setInterval(() => {
        if (clientInstance) {
            clientInstance.updateUserAsync(userData).catch(err => Log.error("Periodic refresh failed", err));
        }
    }, REFRESH_INTERVAL_MS);

    return initializationPromise;
}

export function checkGate(gateName: string): boolean {
    if (!clientInstance) return false;
    return clientInstance.checkGate(gateName);
}

export function getExperimentValue(experimentName: string, key: string, defaultValue: any): any {
    if (!clientInstance) return defaultValue;
    const experiment = clientInstance.getExperiment(experimentName);
    return experiment?.get(key, defaultValue) ?? defaultValue;
}
