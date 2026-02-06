/**
 * File: src/services/statsig/StatsigManager.ts
 * Role: Higher-level manager for Statsig feature gates and experiments.
 */

import { StatsigClientBase, StatsigUser, StatsigClientOptions } from "./StatsigClientBase.js";
import { Log, Diagnostics } from "./StatsigService.js";

export class StatsigClient extends StatsigClientBase {
    private _values: any = null;

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

    async _initializeAsyncImpl(options?: any): Promise<any> {
        Log.info("Starting Statsig initialization...");
        this._setStatus("Initializing", null);
        return this.updateUserAsync(this._user, options);
    }

    async updateUserAsync(newUser: StatsigUser, _options?: any): Promise<any> {
        this._user = newUser;
        Diagnostics.markInitOverallStart(this._sdkKey);
        this._setStatus("Loading", null);

        try {
            Diagnostics.markInitNetworkReqStart(this._sdkKey, { url: "/initialize" });

            // In a real implementation, this would be a network fetch.
            // For now, we simulate a successful fetch with mock evaluations.
            await new Promise(resolve => setTimeout(resolve, 500));

            this._values = {
                feature_gates: {
                    // "tengu_tool_pear" is the "Pair Programmer" / Swarm mode gate.
                    // In the original code (chunk417), this is gated and likely defaults to false unless
                    // explicitly enabled for the user. We force it to TRUE here to enable Swarm features.
                    "tengu_tool_pear": { value: true, rule_id: "default" },

                    // "tengu_lsp_enabled" controls the Language Server Protocol features.
                    // Providing this explicitly ensures LSP features are active.
                    "tengu_lsp_enabled": { value: true, rule_id: "default" },

                    // Other gates found in original code (chunk417, chunk1169) but not forced here:
                    // - "tengu_disable_bypass_permissions_mode": Defaults false (good). If true, disables "bypass permissions".
                    // - "tengu_plan_mode": Related to agent planning capabilities.
                    // - "tengu_permission_explainer": Likely for explaining why permissions are needed.
                    // - "tengu_version_check": Controls version checking behavior.
                },
                dynamic_configs: {},
                layer_configs: {},
                experiments: {
                    "tool_use_examples": {
                        value: { enabled: true },
                        rule_id: "exp_rule_1"
                    }
                }
            };

            Diagnostics.markInitNetworkReqEnd(this._sdkKey, { success: true });
            Diagnostics.markInitProcessStart(this._sdkKey);

            this._setStatus("Ready", this._values);

            Diagnostics.markInitProcessEnd(this._sdkKey, true);
            Diagnostics.markInitOverallEnd(this._sdkKey, true, this._values);

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
        if (this.loadingStatus !== "Ready" || !this._values) return false;

        return this._memoize("gate", (name) => {
            const gate = this._values.feature_gates[name];
            return gate ? gate.value === true : false;
        })(gateName);
    }

    getExperiment(experimentName: string) {
        return {
            get: (key: string, defaultValue: any) => {
                if (this.loadingStatus !== "Ready" || !this._values) return defaultValue;

                return this._memoize("experiment", (name) => {
                    const exp = this._values.experiments[name];
                    if (exp && exp.value && key in exp.value) {
                        return exp.value[key];
                    }
                    return defaultValue;
                })(experimentName);
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

export function logout() {
    clientInstance = null;
    initializationPromise = null;
    Log.info("Statsig state cleared on logout.");
}
