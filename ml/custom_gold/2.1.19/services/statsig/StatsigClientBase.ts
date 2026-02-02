/**
 * File: src/services/statsig/StatsigClientBase.ts
 * Role: Base class for Statsig clients, providing core event emitting and status logic.
 */

import { Log, getStatsigGlobal } from './StatsigService.js';

export interface StatsigUser {
    userId: string;
    email?: string;
    custom?: Record<string, string | number | boolean>;
    [key: string]: any;
}

export type LoadingStatus = "Uninitialized" | "Initializing" | "Loading" | "Ready" | "error";

export interface StatsigClientOptions {
    logLevel?: number;
    disableStorage?: boolean;
    disableLogging?: boolean;
    initialSessionID?: string;
    [key: string]: any;
}

export abstract class StatsigClientBase {
    public loadingStatus: LoadingStatus = "Uninitialized";
    protected _sdkKey: string;
    protected _user: StatsigUser;
    protected options: StatsigClientOptions;
    protected _listeners: Record<string, Function[]> = {};
    protected _memoCache: Record<string, any> = {};
    protected _initializePromise: Promise<any> | null = null;

    constructor(sdkKey: string, user: StatsigUser, options: StatsigClientOptions = {}) {
        this._sdkKey = sdkKey;
        this._user = user;
        this.options = options;

        if (options.logLevel != null) {
            Log.level = options.logLevel;
        }

        this._registerInstance();
    }

    private _registerInstance() {
        const statsig = getStatsigGlobal();
        statsig.instances = statsig.instances || {};
        statsig.instances[this._sdkKey] = this;
        if (!statsig.firstInstance) {
            statsig.firstInstance = this;
        }
    }

    public on(eventName: string, callback: Function) {
        if (!this._listeners[eventName]) {
            this._listeners[eventName] = [];
        }
        this._listeners[eventName].push(callback);
    }

    public off(eventName: string, callback: Function) {
        if (this._listeners[eventName]) {
            this._listeners[eventName] = this._listeners[eventName].filter(cb => cb !== callback);
        }
    }

    /**
     * Internal emitter for Statsig events.
     */
    protected $emt(event: { name: string;[key: string]: any }) {
        const emitters = [
            ...(this._listeners[event.name] || []),
            ...(this._listeners["*"] || [])
        ];

        emitters.forEach(cb => {
            try {
                cb(event);
            } catch (e) {
                Log.error(`Error in Statsig event listener for ${event.name}:`, e);
            }
        });
    }

    protected _setStatus(status: LoadingStatus, values: any) {
        this.loadingStatus = status;
        this._memoCache = {};
        this.$emt({
            name: "values_updated",
            status,
            values
        });
    }

    /**
     * Evaluation memoization helper.
     */
    protected _memoize<T>(prefix: string, fn: (key: string, options?: any) => T): (key: string, options?: any) => T {
        return (key: string, options?: any) => {
            if (this.options.disableEvaluationMemoization) {
                return fn(key, options);
            }
            const memoKey = `${prefix}:${key}:${JSON.stringify(options || {})}`;
            if (!(memoKey in this._memoCache)) {
                // Clear cache if too large (gold reference uses 3000)
                if (Object.keys(this._memoCache).length >= 3000) {
                    this._memoCache = {};
                }
                this._memoCache[memoKey] = fn(key, options);
            }
            return this._memoCache[memoKey];
        };
    }

    public abstract initializeAsync(options?: any): Promise<any>;
    public abstract updateUserAsync(newUser: StatsigUser, options?: any): Promise<any>;
}
