/**
 * File: src/services/auth/msalConfig.ts
 * Role: Configuration and defaults for the MSAL (Microsoft Authentication Library) integration.
 */

/**
 * Default logger configuration for MSAL.
 */
export const DEFAULT_LOGGER_OPTIONS = {
    loggerCallback: () => { },
    piiLoggingEnabled: false,
    logLevel: "Info",
    correlationId: ""
};

/**
 * Default system options for MSAL.
 */
export const DEFAULT_SYSTEM_OPTIONS = {
    tokenRenewalOffsetSeconds: 300,
    preventCorsPreflight: false
};

/**
 * Default cache options for MSAL.
 */
export const DEFAULT_CACHE_OPTIONS = {
    claimsBasedCachingEnabled: false
};

/**
 * Default library information.
 */
export const DEFAULT_LIBRARY_INFO = {
    sku: "msal.js",
    version: "1.2.3", // Example version
    cpu: process.arch || "",
    os: process.platform || ""
};

/**
 * MSAL Telemetry Tracker Class (Stub).
 */
export class TelemetryTracker {
    generateId(): string {
        return Math.random().toString(36).substring(2, 15);
    }

    startMeasurement(name: string, correlationId?: string) {
        return {
            end: () => null,
            discard: () => { },
            add: () => { },
            increment: () => { },
            event: {
                eventId: this.generateId(),
                status: "InProgress",
                name,
                startTimeMs: Date.now(),
                correlationId: correlationId || ""
            }
        };
    }
}

/**
 * Creates a complete MSAL configuration object by merging defaults with overrides.
 */
export function createMSALConfiguration(overrides: any = {}): any {
    return {
        authOptions: {
            clientCapabilities: [],
            azureCloudOptions: {
                azureCloudInstance: "None",
                tenant: "common"
            },
            skipAuthorityMetadataCache: false,
            instanceAware: false,
            encodeExtraQueryParams: false,
            ...overrides.authOptions
        },
        systemOptions: {
            ...DEFAULT_SYSTEM_OPTIONS,
            ...overrides.systemOptions
        },
        loggerOptions: {
            ...DEFAULT_LOGGER_OPTIONS,
            ...overrides.loggerOptions
        },
        cacheOptions: {
            ...DEFAULT_CACHE_OPTIONS,
            ...overrides.cacheOptions
        },
        libraryInfo: {
            ...DEFAULT_LIBRARY_INFO,
            ...overrides.libraryInfo
        },
        ...overrides
    };
}
