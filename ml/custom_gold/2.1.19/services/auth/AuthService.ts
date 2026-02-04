/**
 * File: src/services/auth/AuthService.ts
 * Role: Orchestrates API key and OAuth authentication headers.
 */

import axios from 'axios';
import { ApiKeyManager } from './ApiKeyManager.js';
import { OAuthService } from './OAuthService.js';
import { PROFILE_URL, OAUTH_BETA_HEADER } from '../../constants/product.js';

interface ProfileResponse {
    account: {
        display_name: string;
    };
    organization: {
        organization_type: string;
        rate_limit_tier: string;
        has_extra_usage_enabled: boolean;
    };
}

let cachedProfile: { plan: string, displayName?: string } | null = null;
let lastFetchTime = 0;
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Maps organization type to a user-friendly plan name.
 */
/**
 * Maps organization type and rate limit tier to a user-friendly plan name.
 */
function mapPlanName(orgType: string, tier: string): string {
    const basePlan = (() => {
        switch (orgType) {
            case 'claude_max': return 'Max';
            case 'claude_pro': return 'Pro';
            case 'claude_team': return 'Team';
            case 'claude_enterprise': return 'Enterprise';
            default: return 'Free';
        }
    })();

    if (tier && tier !== 'default') {
        // Clean up tier name (e.g., 'default_claude_max_20x' -> '20x')
        // Removing prefix commonly found in tier names
        const prefix = `default_${orgType}_`;
        let shortTier = tier;
        if (shortTier.startsWith(prefix)) {
            shortTier = shortTier.substring(prefix.length);
        }
        shortTier = shortTier.replace(/_/g, ' ');
        return `${basePlan} Plan (${shortTier})`;
    }

    return `${basePlan} Plan`;
}

/**
 * Fetches the user profile using the OAuth access token.
 */
async function fetchProfile(accessToken: string): Promise<{ plan: string, displayName?: string } | null> {
    const now = Date.now();
    if (cachedProfile && now - lastFetchTime < CACHE_TTL) {
        return cachedProfile;
    }

    const doFetch = async (token: string) => {
        const data = await OAuthService.fetchProfile(token);
        const { organization, account } = data;
        return {
            plan: mapPlanName(organization.organization_type, organization.rate_limit_tier),
            displayName: account.display_name
        };
    };

    try {
        cachedProfile = await doFetch(accessToken);
        lastFetchTime = now;
        return cachedProfile;
    } catch (error: any) {
        // Retry on 401 (Unauthorized)
        if (error.response?.status === 401) {
            try {
                // First try a non-forced token (may have been refreshed by another call)
                let newToken = await OAuthService.getValidToken(false);

                // If the token is the same as the one that failed, then force a refresh
                if (newToken === accessToken) {
                    newToken = await OAuthService.getValidToken(true);
                }

                if (newToken && newToken !== accessToken) {
                    cachedProfile = await doFetch(newToken);
                    lastFetchTime = now;
                    return cachedProfile;
                }
            } catch (retryError) {
                // Fall through to error logging
            }
        }

        console.error('Failed to fetch user profile:', error instanceof Error ? error.message : String(error));
        return null;
    }
}

/**
 * Returns the appropriate authentication headers for Anthropic API requests.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
    // 1. Try OAuth first if available
    const oauthToken = await OAuthService.getValidToken();
    if (oauthToken) {
        return {
            "Authorization": `Bearer ${oauthToken}`,
            "anthropic-beta": OAUTH_BETA_HEADER,
            "anthropic-version": "2023-06-01"
        };
    }

    // 2. Fallback to API Key Manager
    const apiKey = await ApiKeyManager.getApiKey();
    if (apiKey) {
        return {
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01"
        };
    }

    return {};
}

/**
 * Checks if the user is authenticated.
 */
export async function isAuthenticated(): Promise<boolean> {
    const headers = await getAuthHeaders();
    return Object.keys(headers).length > 0;
}

/**
 * Returns the authentication type and plan description.
 */
export async function getAuthDetails(): Promise<{ type: 'oauth' | 'apikey' | 'none', plan: string }> {
    const oauthToken = await OAuthService.getValidToken();
    if (oauthToken) {
        const session = (OAuthService as any).session; // Accessing internal for details if possible
        const profile = await fetchProfile(oauthToken);
        return {
            type: 'oauth',
            plan: profile?.plan || 'OAuth Account'
        };
    }

    const apiKey = await ApiKeyManager.getApiKey();
    if (apiKey) {
        return { type: 'apikey', plan: 'API Pay-as-you-go' };
    }

    return { type: 'none', plan: 'Not Authenticated' };
}
/**
 * Clears the authentication cache and session details.
 */
export async function logout(): Promise<void> {
    cachedProfile = null;
    lastFetchTime = 0;
    // OAuthService.logout() is called separately by the command dispatcher
}
