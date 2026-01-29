/**
 * File: src/services/telemetry/OtlpConfig.ts
 * Role: OTLP (OpenTelemetry) Configuration and Export delegates.
 */

import { getStringFromEnv, getNumberFromEnv, parseKeyPairsIntoRecord } from '../../utils/shared/runtimeAndEnv.js';
// diag is used in the original JS but not actually used in the simplified version I saw. 
// I'll leave the import if it was there or skip if unused. 
// The JS had import { diag } from ...; but didn't use it in the displayed lines.
// I'll skip it for now unless I see usage.
// Actually, I'll import it just in case.
import { diag } from '../../utils/shared/runtime.js';

export interface OtlpConfig {
    timeoutMillis: number;
    compression: 'gzip' | 'none';
    url?: string;
    headers?: Record<string, string>;
}

/**
 * Gets shared OTLP configuration from environment variables.
 */
export function getSharedRetryingConfig(): OtlpConfig {
    const timeout = getNumberFromEnv("OTEL_EXPORTER_OTLP_TIMEOUT") || 10000;
    const compression = getStringFromEnv("OTEL_EXPORTER_OTLP_COMPRESSION") || 'none';

    return {
        timeoutMillis: timeout,
        compression: compression === 'gzip' ? 'gzip' : 'none'
    };
}

/**
 * Returns a full HTTP configuration for OTLP exporters.
 */
export function getOtlpHttpConfig(type = 'TRACES'): OtlpConfig {
    const shared = getSharedRetryingConfig();
    const endpoint = getStringFromEnv(`OTEL_EXPORTER_OTLP_${type}_ENDPOINT`)
        || getStringFromEnv("OTEL_EXPORTER_OTLP_ENDPOINT");

    const headerStr = getStringFromEnv(`OTEL_EXPORTER_OTLP_${type}_HEADERS`)
        || getStringFromEnv("OTEL_EXPORTER_OTLP_HEADERS");

    const headers = parseKeyPairsIntoRecord(headerStr);

    return {
        ...shared,
        url: endpoint,
        headers: headers
    };
}
