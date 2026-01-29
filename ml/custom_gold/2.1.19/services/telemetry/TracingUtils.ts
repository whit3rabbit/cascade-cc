/**
 * File: src/services/telemetry/TracingUtils.ts
 * Role: Utilities for distributed tracing and Sentry-compatible baggage headers.
 */

const MAX_BAGGAGE_LENGTH = 8192;
const SENTRY_PREFIX = "sentry-";

/**
 * Parses a baggage header string into dynamic sampling context.
 */
export function baggageHeaderToDynamicSamplingContext(baggageHeader: string | string[]): Record<string, string> | undefined {
    if (!baggageHeader) return undefined;

    const pairs: Record<string, string> = {};
    const headers = Array.isArray(baggageHeader) ? baggageHeader : [baggageHeader];

    for (const header of headers) {
        const items = header.split(",").map(i => i.trim());
        for (const item of items) {
            const [key, value] = item.split("=").map(i => i.trim());
            if (key && value && key.startsWith(SENTRY_PREFIX)) {
                pairs[key.slice(SENTRY_PREFIX.length)] = value;
            }
        }
    }

    return Object.keys(pairs).length > 0 ? pairs : undefined;
}

/**
 * Serializes dynamic sampling context into a Sentry-compatible baggage header.
 */
export function dynamicSamplingContextToSentryBaggageHeader(dsc: Record<string, string>): string | undefined {
    if (!dsc) return undefined;

    const parts: string[] = [];
    for (const [key, value] of Object.entries(dsc)) {
        if (value) {
            const entry = `${encodeURIComponent(SENTRY_PREFIX + key)}=${encodeURIComponent(value)}`;
            if (parts.join(",").length + entry.length + 1 > MAX_BAGGAGE_LENGTH) {
                console.warn(`[Tracing] Baggage header exceeded size limit, skipping key: ${key}`);
                continue;
            }
            parts.push(entry);
        }
    }

    return parts.length > 0 ? parts.join(",") : undefined;
}
