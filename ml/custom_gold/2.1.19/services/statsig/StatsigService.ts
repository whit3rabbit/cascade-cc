/**
 * File: src/services/statsig/StatsigService.ts
 * Role: Statsig Integration and Diagnostics
 */

// Placeholder stubs for Statsig classes based on deobfuscated code
export const LogLevel = {
    StatsigLogLevelNone: 0,
    Error: 1,
    Warn: 2,
    Info: 3,
    Debug: 4,
};

export class Log {
    static level = LogLevel.Warn;

    static info(...args: any[]) {
        if (Log.level >= LogLevel.Info) {
            console.info("[Statsig Info]", ...args);
        }
    }
    static debug(...args: any[]) {
        if (Log.level >= LogLevel.Debug) {
            console.debug("[Statsig Debug]", ...args);
        }
    }
    static warn(...args: any[]) {
        if (Log.level >= LogLevel.Warn) {
            console.warn("[Statsig Warn]", ...args);
        }
    }
    static error(...args: any[]) {
        if (Log.level >= LogLevel.Error) {
            console.error("[Statsig Error]", ...args);
        }
    }
}

export const Diagnostics = {
    markers: new Map<string, any[]>(),

    markInitOverallStart(key: string) {
        this._addMarker(key, { action: "start", step: "overall" });
    },
    markInitOverallEnd(key: string, success: boolean, evaluationDetails: any) {
        this._addMarker(key, {
            action: "end",
            step: "overall",
            success,
            error: success ? undefined : { name: "InitializeError", message: "Failed to initialize" },
            evaluationDetails
        });
    },
    markInitNetworkReqStart(key: string, networkReqData: any) {
        this._addMarker(key, { action: "start", step: "network_request", ...networkReqData });
    },
    markInitNetworkReqEnd(key: string, networkReqData: any) {
        this._addMarker(key, { action: "end", step: "network_request", ...networkReqData });
    },

    _addMarker(key: string, data: any) {
        const list = this.markers.get(key) || [];
        list.push({ ...data, timestamp: Date.now() });
        this.markers.set(key, list);
    },

    clearMarkers(key: string) {
        this.markers.delete(key);
    },

    getDiagnosticsData(response: Response | undefined, attempt: number, isDelta: boolean, errorData: any) {
        return {
            success: response?.ok === true,
            statusCode: response?.status,
            sdkRegion: response?.headers?.get("x-statsig-region"),
            isDelta: isDelta ? true : undefined,
            attempt,
            error: errorData ? { name: errorData.name, message: errorData.message, code: errorData.code } : undefined
        };
    }
};

// Global Statsig reference setup
if (typeof window !== "undefined") {
    (window as any).__STATSIG__ = (window as any).__STATSIG__ || {};
} else if (typeof global !== "undefined") {
    (global as any).__STATSIG__ = (global as any).__STATSIG__ || {};
}
