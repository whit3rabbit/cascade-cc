/**
 * File: src/utils/networkUtils.ts
 * Role: Aggregated networking and authentication utilities.
 */

export * from "../services/auth/azureCredentials.js";
export * from "./shared/fixtureUtils.js";

/**
 * Standardized logger configuration for SDKs (e.g. Anthropic).
 */
export const SDK_LOGGER = {
    error: (message: string, ...args: any[]) => console.error("[SDK ERROR]", message, ...args),
    warn: (message: string, ...args: any[]) => console.error("[SDK WARN]", message, ...args),
    info: (message: string, ...args: any[]) => console.info("[SDK INFO]", message, ...args),
};
