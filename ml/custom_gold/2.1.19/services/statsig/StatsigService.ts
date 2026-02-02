/**
 * File: src/services/statsig/StatsigService.ts
 * Role: Statsig Integration and Diagnostics
 */

export const LogLevel = {
    StatsigLogLevelNone: 0,
    Error: 1,
    Warn: 2,
    Info: 3,
    Debug: 4,
};

const DEBUG_PREFIX = " DEBUG ";
const INFO_PREFIX = "  INFO ";
const WARN_PREFIX = "  WARN ";
const ERROR_PREFIX = " ERROR ";

function formatLogArgs(args: any[]) {
    args.unshift("[Statsig]");
    return args;
}

export class Log {
    static level = LogLevel.Warn;

    static info(...args: any[]) {
        if (Log.level >= LogLevel.Info) {
            console.info(INFO_PREFIX, ...formatLogArgs(args));
        }
    }
    static debug(...args: any[]) {
        if (Log.level >= LogLevel.Debug) {
            console.debug(DEBUG_PREFIX, ...formatLogArgs(args));
        }
    }
    static warn(...args: any[]) {
        if (Log.level >= LogLevel.Warn) {
            console.warn(WARN_PREFIX, ...formatLogArgs(args));
        }
    }
    static error(...args: any[]) {
        if (Log.level >= LogLevel.Error) {
            console.error(ERROR_PREFIX, ...formatLogArgs(args));
        }
    }
}

const markers = new Map<string, any[]>();
const ACTION_START = "start";
const ACTION_END = "end";
const STEP_OVERALL = "overall";
const STEP_NETWORK_REQUEST = "network_request";
const STEP_PROCESS = "process";

export const Diagnostics = {
    _getMarkers(key: string) {
        return markers.get(key);
    },

    markInitOverallStart(key: string) {
        this._addMarker(key, { action: ACTION_START, step: STEP_OVERALL });
    },

    markInitOverallEnd(key: string, success: boolean, evaluationDetails: any) {
        this._addMarker(key, {
            action: ACTION_END,
            step: STEP_OVERALL,
            success,
            error: success ? undefined : { name: "InitializeError", message: "Failed to initialize" },
            evaluationDetails
        });
    },

    markInitNetworkReqStart(key: string, networkReqData: any) {
        this._addMarker(key, { action: ACTION_START, step: STEP_NETWORK_REQUEST, ...networkReqData });
    },

    markInitNetworkReqEnd(key: string, networkReqData: any) {
        this._addMarker(key, { action: ACTION_END, step: STEP_NETWORK_REQUEST, ...networkReqData });
    },

    markInitProcessStart(key: string) {
        this._addMarker(key, { action: ACTION_START, step: "initialize", subStep: STEP_PROCESS });
    },

    markInitProcessEnd(key: string, success: boolean) {
        this._addMarker(key, { action: ACTION_END, step: "initialize", subStep: STEP_PROCESS, success });
    },

    _addMarker(key: string, data: any) {
        const Y = markers.get(key) ?? [];
        Y.push({ ...data, timestamp: Date.now() });
        markers.set(key, Y);
    },

    clearMarkers(key: string) {
        markers.delete(key);
    },

    getDiagnosticsData(response: any, attempt: number, isDelta: boolean, errorData: any) {
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
const STATSIG_GLOBAL = "__STATSIG__";
const globalRef: any = typeof window !== "undefined" ? window : (typeof global !== "undefined" ? global : (typeof globalThis !== "undefined" ? globalThis : {}));

if (!globalRef[STATSIG_GLOBAL]) {
    globalRef[STATSIG_GLOBAL] = {};
}

export const getStatsigGlobal = () => globalRef[STATSIG_GLOBAL];
