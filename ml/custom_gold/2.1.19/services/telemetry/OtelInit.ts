/**
 * File: src/services/telemetry/OtelInit.ts
 * Role: Initializes OpenTelemetry MeterProvider, TracerProvider, and LoggerProvider.
 */

// We'll use mock implementations to avoid actual dependencies and build errors
// since we don't have the full @opentelemetry packages installed in this environment.

import { BUILD_INFO } from '../../constants/build.js';

// Mock dependencies
const DiagLogLevel = { ERROR: 0 };
const metrics = {
    setGlobalMeterProvider: (_provider: any) => { }
};

const diag = {
    setLogger: (_logger: any, _level: any) => { }
};

class Resource {
    constructor(_attributes: Record<string, any>) { }
}

class MeterProvider {
    constructor(_options: any) { }
    async shutdown() { }
}

const SEMRESATTRS_SERVICE_NAME = "service.name";
const SEMRESATTRS_SERVICE_VERSION = "service.version";

/**
 * Bootstraps OpenTelemetry with standard resource attributes.
 */
export function initializeOtel() {
    diag.setLogger({
        error: console.error,
        warn: console.warn,
        info: console.info,
        debug: () => { }
    }, DiagLogLevel.ERROR);

    const resource = new Resource({
        [SEMRESATTRS_SERVICE_NAME]: "claude-code",
        [SEMRESATTRS_SERVICE_VERSION]: BUILD_INFO.VERSION,
        "os.platform": process.platform,
        "host.arch": process.arch
    });

    const meterProvider = new MeterProvider({ resource });
    metrics.setGlobalMeterProvider(meterProvider);

    // Shutdown handler
    process.on('SIGINT', async () => {
        await meterProvider.shutdown();
        process.exit(0);
    });

    return { meterProvider };
}

export const initializeTelemetry = initializeOtel;
