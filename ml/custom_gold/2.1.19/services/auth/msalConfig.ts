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
 * MSAL Telemetry Tracker Class.
 */
export class TelemetryTracker {
    private measurements = new Map<string, { startTime: number; correlationId: string }>();

    generateId(): string {
        return Math.random().toString(36).substring(2, 15);
    }

    startMeasurement(name: string, correlationId?: string) {
        const eventId = this.generateId();
        const startTimeMs = Date.now();
        const corrId = correlationId || "";

        this.measurements.set(eventId, { startTime: startTimeMs, correlationId: corrId });

        return {
            end: () => {
                const info = this.measurements.get(eventId);
                if (info) {
                    const duration = Date.now() - info.startTime;
                    // In a real impl, this would log to a telemetry provider
                    this.measurements.delete(eventId);
                    return duration;
                }
                return null;
            },
            discard: () => {
                this.measurements.delete(eventId);
            },
            add: (key: string, value: any) => {
                // Add attribute to measurement
            },
            increment: (key: string) => {
                // Increment counter
            },
            event: {
                eventId,
                status: "InProgress",
                name,
                startTimeMs,
                correlationId: corrId
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
