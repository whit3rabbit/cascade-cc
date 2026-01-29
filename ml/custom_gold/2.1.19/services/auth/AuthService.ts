/**
 * File: src/services/auth/AuthService.ts
 * Role: Orchestrates API key and OAuth authentication headers.
 */

import { ApiKeyManager } from './ApiKeyManager.js';
import { OAuthService } from './OAuthService.js';

/**
 * Returns the appropriate authentication headers for Anthropic API requests.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
    // 1. Try OAuth first if available
    const oauthToken = await OAuthService.getValidToken();
    if (oauthToken) {
        return {
            "Authorization": `Bearer ${oauthToken}`,
            "anthropic-beta": "claude-3-5-sonnet-20241022" // Example beta header
        };
    }

    // 2. Fallback to API Key
    const apiKey = ApiKeyManager.getApiKey();
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
