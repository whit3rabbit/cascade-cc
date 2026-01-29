/**
 * File: src/utils/auth/authHelpers.ts
 * Role: Helper functions for authentication and feature flag checks.
 */

export interface AccessTokenData {
    accessToken: string | null;
    scopes: string[];
    subscriptionType: string;
}

/**
 * Checks if the current build/environment is first-party.
 */
export function isFirstParty(): boolean {
    // Stub implementation; in a real app this might check process.env or a build constant.
    return true;
}

/**
 * Checks if a specific feature is enabled.
 */
export function isFeatureEnabled(_featureName?: string): boolean {
    return true;
}

/**
 * Retrieves the current access token data.
 */
export function getAccessTokenData(): AccessTokenData {
    return {
        accessToken: null,
        scopes: [],
        subscriptionType: "none"
    };
}

/**
 * Simple helper for API key options.
 */
export function getApiKeyHelper(_options?: any): { key: string | null } {
    return { key: null };
}

// --- Aliases for compatibility ---
export {
    isFirstParty as cK,
    isFeatureEnabled as VeA,
    getAccessTokenData as FK,
    getApiKeyHelper as z_
};
