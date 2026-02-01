/**
 * File: src/services/auth/MsalAuthService.ts
 * Role: Core MSAL-based authentication service for managing tokens and account state.
 */

import { PublicClientApplication, Configuration, LogLevel, AuthorizationUrlRequest, AuthorizationCodeRequest } from '@azure/msal-node';
import { createMSALConfiguration } from './msalConfig.js';
import { Account } from './AccountManager.js';
import * as http from 'http';
import open from 'open';

export interface TokenRequest {
    account?: Account;
    scopes?: string[];
    loginHint?: string;
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
    private client: PublicClientApplication;
    private config: any;

    constructor(configOverrides: any = {}) {
        const rawConfig = createMSALConfiguration(configOverrides);

        const msalConfig: Configuration = {
            auth: {
                clientId: rawConfig.authOptions?.clientId || "claude-code-cli-id",
                authority: rawConfig.authOptions?.authority || "https://login.microsoftonline.com/common",
                clientSecret: rawConfig.authOptions?.clientSecret,
            },
            system: {
                loggerOptions: {
                    loggerCallback(loglevel: LogLevel, message: string, containsPii: boolean) {
                        // console.log(message);
                    },
                    piiLoggingEnabled: false,
                    logLevel: LogLevel.Info,
                }
            },
            cache: rawConfig.cacheOptions
        };

        this.client = new PublicClientApplication(msalConfig);
    }

    /**
     * Acquires a token silently from the cache.
     */
    async acquireTokenSilent(request: TokenRequest): Promise<AuthenticationResult> {
        try {
            const result = await this.client.acquireTokenSilent({
                account: request.account as any,
                scopes: request.scopes || ["User.Read"],
            });

            if (!result) throw new Error("Silent token acquisition failed");

            return {
                accessToken: result.accessToken,
                idToken: result.idToken,
                expiresOn: result.expiresOn || undefined,
                account: result.account ? {
                    homeAccountId: result.account.homeAccountId,
                    username: result.account.username,
                    environment: result.account.environment,
                    tenantId: result.account.tenantId,
                    localAccountId: result.account.localAccountId
                } : undefined
            };
        } catch (error) {
            // silent failure is expected, caller should fall back to interactive
            throw error;
        }
    }

    /**
     * Acquires a token interactively via browser (Authorization Code Flow).
     */
    async acquireTokenInteractive(request: TokenRequest): Promise<AuthenticationResult> {
        return new Promise((resolve, reject) => {
            const server = http.createServer(async (req, res) => {
                const url = new URL(req.url!, `http://${req.headers.host}`);
                const code = url.searchParams.get('code');

                if (code) {
                    res.writeHead(200, { 'Content-Type': 'text/html' });
                    res.end('<h1>Authentication successful! You can close this window.</h1>');
                    server.close();

                    try {
                        const result = await this.client.acquireTokenByCode({
                            code,
                            scopes: request.scopes || ["User.Read"],
                            redirectUri: "http://localhost:3000/redirect", // Must match registration
                        });

                        resolve({
                            accessToken: result.accessToken,
                            idToken: result.idToken,
                            expiresOn: result.expiresOn || undefined,
                            account: result.account ? {
                                homeAccountId: result.account.homeAccountId,
                                username: result.account.username
                            } : undefined
                        });
                    } catch (err) {
                        reject(err);
                    }
                }
            });

            server.listen(3000, async () => {
                const authUrl = await this.client.getAuthCodeUrl({
                    scopes: request.scopes || ["User.Read"],
                    redirectUri: "http://localhost:3000/redirect",
                    loginHint: request.loginHint
                });
                await open(authUrl);
            });

            // Timeout after 5 minutes
            setTimeout(() => {
                server.close();
                reject(new Error("Timeout waiting for authentication"));
            }, 300000);
        });
    }

    /**
     * Acquires a token using an authorization code (Manual).
     */
    async acquireTokenByCode(request: TokenRequest & { code: string, redirectUri: string }): Promise<AuthenticationResult> {
        const result = await this.client.acquireTokenByCode({
            code: request.code,
            scopes: request.scopes || ["User.Read"],
            redirectUri: request.redirectUri
        });

        return {
            accessToken: result.accessToken,
            idToken: result.idToken,
            expiresOn: result.expiresOn || undefined,
            account: result.account ? {
                homeAccountId: result.account.homeAccountId
            } : undefined
        };
    }

    /**
     * Acquires a token using a refresh token.
     */
    async acquireTokenByRefreshToken(request: TokenRequest): Promise<AuthenticationResult> {
        // MSAL Node handles refresh tokens automatically in acquireTokenSilent
        // But if needed explicitly, we might use the token cache directly or different method
        // For now, we delegate to silent which uses RT
        return this.acquireTokenSilent(request);
    }

    /**
     * Clears the local account and token cache.
     */
    async logout(account?: Account): Promise<void> {
        const cache = this.client.getTokenCache();
        if (account) {
            const msalAccount = await cache.getAccountByHomeId(account.homeAccountId);
            if (msalAccount) {
                await cache.removeAccount(msalAccount);
            }
        }
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
