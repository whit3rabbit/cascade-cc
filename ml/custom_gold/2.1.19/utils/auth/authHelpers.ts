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
 * Checks if the current build/environment is first-party (Anthropic API).
 */
export function isFirstParty(): boolean {
    const bedrock = process.env.CLAUDE_CODE_USE_BEDROCK;
    const vertex = process.env.CLAUDE_CODE_USE_VERTEX;
    const foundry = process.env.CLAUDE_CODE_USE_FOUNDRY;

    // If any of these are set, it's NOT first-party (it's a proxy/partner integration)
    return !(bedrock || vertex || foundry);
}

/**
 * Checks if a specific feature is enabled.
 */
export function isFeatureEnabled(_featureName?: string): boolean {
    const baseUrl = process.env.ANTHROPIC_BASE_URL;
    if (!baseUrl) return true;

    try {
        const host = new URL(baseUrl).host;
        return ["api.anthropic.com"].includes(host);
    } catch {
        return false;
    }
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
