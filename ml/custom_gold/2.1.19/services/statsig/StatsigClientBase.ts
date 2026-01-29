/**
 * File: src/services/statsig/StatsigClientBase.ts
 * Role: Base class for Statsig clients, providing core evaluation and event logging logic.
 */

// import { Log } from './StatsigService.js';
import { NetworkCore, NetworkCoreOptions } from './NetworkCore.js';

export interface StatsigUser {
    userId: string;
    email?: string;
    custom?: Record<string, string | number | boolean>;
    [key: string]: any;
}

export interface Experiment {
    get(key: string, defaultValue: any): any;
}

export class StatsigClient {
    protected sdkKey: string;
    protected user: StatsigUser;
    protected options: NetworkCoreOptions;
    protected networkCore: NetworkCore;
    protected status: "uninitialized" | "initializing" | "initialized" | "error";
    protected memoCache: Map<string, any>;

    constructor(sdkKey: string, userData: StatsigUser, options: NetworkCoreOptions = {}) {
        this.sdkKey = sdkKey;
        this.user = userData;
        this.options = options;
        this.networkCore = new NetworkCore(options);
        this.status = "uninitialized";
        this.memoCache = new Map();
    }

    /**
     * Initializes the client by fetching configuration from the server.
     */
    async initializeAsync(): Promise<void> {
        this.status = "initializing";
        try {
            // In a real impl, this would fetch 'initialize' data
            this.status = "initialized";
            console.log(`[Statsig] Client initialized for user: ${this.user.userId}`);
        } catch (error) {
            this.status = "error";
            console.error("Statsig initialization failed:", error);
        }
    }

    /**
     * Checks a feature gate.
     */
    checkGate(gateName: string): boolean {
        if (this.status !== "initialized") return false;
        // Mocking gate check
        return true;
    }

    /**
     * Gets an experiment configuration.
     */
    getExperiment(experimentName: string): Experiment {
        return {
            get: (key: string, defaultValue: any) => defaultValue
        };
    }

    /**
     * Updates the current user and refreshes config.
     */
    async updateUserAsync(newUser: StatsigUser): Promise<void> {
        this.user = newUser;
        return this.initializeAsync();
    }
}
