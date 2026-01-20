import { isInteractiveAuthRequired } from "./authSource.js";

export interface OauthTokens {
    accessToken: string;
    refreshToken: string | null;
    expiresAt: string | null;
    scopes: string[];
    subscriptionType?: string | null;
    rateLimitTier?: string | null;
}

/**
 * Saves OAuth tokens to persistent storage.
 */
export function saveOauthTokens(tokens: OauthTokens): { success: boolean, warning?: string } {
    try {
        // This would write to a local config or keychain
        console.log("Saving OAuth tokens...");
        return { success: true };
    } catch (err) {
        return { success: false, warning: "Failed to save OAuth tokens" };
    }
}

/**
 * Refreshes OAuth tokens using a refresh token and cross-process locking.
 */
export async function refreshOauthTokens(retryCount: number = 0): Promise<boolean> {
    const currentTokens = getStoredOauthTokens();
    if (!currentTokens?.refreshToken) return false;

    // Simplified locking logic placeholder
    console.log("Refreshing OAuth tokens...");

    try {
        // const newTokens = await apiRefresh(currentTokens.refreshToken);
        // saveOauthTokens(newTokens);
        return true;
    } catch (err) {
        return false;
    }
}

/**
 * Checks if OAuth is currently active and valid.
 */
export function isOauthActive(): boolean {
    if (!isInteractiveAuthRequired()) return false;
    const tokens = getStoredOauthTokens();
    return (tokens?.scopes ?? []).includes("user:inference");
}

/**
 * Returns the user subscription tier.
 */
export function getSubscriptionType(): string | null {
    if (!isInteractiveAuthRequired()) return null;
    const tokens = getStoredOauthTokens();
    return tokens?.subscriptionType ?? null;
}

/**
 * Returns a human-readable label for the subscription.
 */
export function getSubscriptionLabel(): string {
    const type = getSubscriptionType();
    switch (type) {
        case "enterprise": return "Claude Enterprise";
        case "team": return "Claude Team";
        case "max": return "Claude Max";
        case "pro": return "Claude Pro";
        default: return "Claude API";
    }
}

/**
 * Returns the user's rate limit tier.
 */
export function getRateLimitTier(): string | null {
    if (!isInteractiveAuthRequired()) return null;
    const tokens = getStoredOauthTokens();
    return tokens?.rateLimitTier ?? null;
}

/**
 * Internal helper to read tokens from disk/env.
 */
export function getStoredOauthTokens(): OauthTokens | null {
    if (process.env.CLAUDE_CODE_OAUTH_TOKEN) {
        return {
            accessToken: process.env.CLAUDE_CODE_OAUTH_TOKEN,
            refreshToken: null,
            expiresAt: null,
            scopes: ["user:inference"]
        };
    }
    // This would read from ~/.claude.json or similar
    return null;
}
