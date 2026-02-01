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
function mapPlanName(orgType: string): string {
    switch (orgType) {
        case 'claude_max': return 'Max Plan';
        case 'claude_pro': return 'Pro Plan';
        case 'claude_team': return 'Team Plan';
        case 'claude_enterprise': return 'Enterprise Plan';
        default: return 'Free Plan';
    }
}

/**
 * Fetches the user profile using the OAuth access token.
 */
async function fetchProfile(accessToken: string): Promise<{ plan: string, displayName?: string } | null> {
    const now = Date.now();
    if (cachedProfile && now - lastFetchTime < CACHE_TTL) {
        return cachedProfile;
    }

    try {
        const response = await axios.get<ProfileResponse>(PROFILE_URL, {
            headers: {
                Authorization: `Bearer ${accessToken}`,
                'anthropic-beta': OAUTH_BETA_HEADER
            }
        });

        const { organization, account } = response.data;
        cachedProfile = {
            plan: mapPlanName(organization.organization_type),
            displayName: account.display_name
        };
        lastFetchTime = now;
        return cachedProfile;
    } catch (error) {
        console.error('Failed to fetch user profile:', error instanceof Error ? error.message : String(error));
        return null;
    }
}

/**
 * Returns the appropriate authentication headers for Anthropic API requests.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
    // 0. Check for custom Auth Token
    const customToken = process.env.ANTHROPIC_AUTH_TOKEN;
    if (customToken) {
        return {
            "Authorization": `Bearer ${customToken}`
        };
    }

    // 1. Try OAuth first if available
    const oauthToken = await OAuthService.getValidToken();
    if (oauthToken) {
        return {
            "Authorization": `Bearer ${oauthToken}`,
            "anthropic-beta": OAUTH_BETA_HEADER
        };
    }

    // 2. Fallback to API Key
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
        const profile = await fetchProfile(oauthToken);
        return {
            type: 'oauth',
            plan: profile?.plan || 'Pro Plan'
        };
    }

    const apiKey = await ApiKeyManager.getApiKey();
    if (apiKey) {
        return { type: 'apikey', plan: 'API Pay-as-you-go' };
    }

    return { type: 'none', plan: 'Not Authenticated' };
}
