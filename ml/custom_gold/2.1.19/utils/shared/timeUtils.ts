/**
 * File: src/utils/shared/timeUtils.ts
 * Role: High-precision timing and timestamp utilities.
 */

const SECONDS_IN_MS = 1000;

/**
 * Returns a timestamp in seconds with millisecond precision.
 */
export function dateTimestampInSeconds(): number {
    return Date.now() / SECONDS_IN_MS;
}

/**
 * Returns a high-precision timestamp in seconds using performance.now() if available.
 */
export function timestampInSeconds(): number {
    // If performance.now() is available via globalThis (Node.js or Browser)
    if (typeof performance !== 'undefined' && performance.now) {
        return (performance.timeOrigin + performance.now()) / SECONDS_IN_MS;
    }
    return dateTimestampInSeconds();
}

/**
 * Common sleep/delay utility.
 */
export function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}
