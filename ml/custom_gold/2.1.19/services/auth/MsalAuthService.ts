/**
 * File: src/services/auth/MsalAuthService.ts
 * Role: Core MSAL-based authentication service for managing tokens and account state.
 */

import { createMSALConfiguration } from './msalConfig.js';
import { Account } from './AccountManager.js';

export interface TokenRequest {
    account?: Account;
    scopes?: string[];
    [key: string]: any;
}

export interface AuthenticationResult {
    accessToken: string;
    idToken?: string;
    expiresOn?: Date;
    account?: Account | { homeAccountId: string };
}

/**
 * Service for interfacing with Microsoft Authentication Library (MSAL).
 */
export class MsalAuthService {
    private config: any;
    private logger: any;
    private cache: any;

    constructor(configOverrides: any = {}) {
        this.config = createMSALConfiguration(configOverrides);

        // In a real implementation, we'd initialize the MSAL PublicClientApplication here.
        this.logger = this.config.loggerOptions;
        this.cache = this.config.cacheOptions;
    }

    /**
     * Acquires a token silently from the cache.
     */
    async acquireTokenSilent(request: TokenRequest): Promise<AuthenticationResult> {
        console.log("[MsalAuthService] acquireTokenSilent", request);
        // Stub implementation
        return {
            accessToken: "mock-access-token",
            idToken: "mock-id-token",
            expiresOn: new Date(Date.now() + 3600 * 1000),
            account: request.account
        };
    }

    /**
     * Acquires a token using an authorization code.
     */
    async acquireTokenByCode(request: TokenRequest): Promise<AuthenticationResult> {
        console.log("[MsalAuthService] acquireTokenByCode", request);
        return {
            accessToken: "mock-access-token",
            account: { homeAccountId: "mock-home-id" }
        };
    }

    /**
     * Acquires a token using a refresh token.
     */
    async acquireTokenByRefreshToken(request: TokenRequest): Promise<AuthenticationResult> {
        console.log("[MsalAuthService] acquireTokenByRefreshToken", request);
        return {
            accessToken: "mock-access-token"
        };
    }

    /**
     * Clears the local account and token cache.
     */
    logout(): void {
        console.log("[MsalAuthService] Logging out...");
        // Clear storage logic
    }
}

// Global state for Statsig/Auth integration (legacy support)
const statsigStorage: { oauthAccount: any; accessToken: any } = {
    oauthAccount: null,
    accessToken: null,
};

export function getStatsigStorage() {
    return statsigStorage;
}

export function setStatsigStorage(updates: Partial<typeof statsigStorage>) {
    Object.assign(statsigStorage, updates);
}
