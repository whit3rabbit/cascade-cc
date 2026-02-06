/**
 * File: src/services/auth/MsalAuthService.ts
 * Role: Core MSAL-based authentication service for managing tokens and account state.
 */

import { PublicClientApplication, Configuration, LogLevel } from '@azure/msal-node';
import { createMSALConfiguration } from './msalConfig.js';
import { Account } from './AccountManager.js';
import * as http from 'http';
import open from 'open';
import { HTTP_PROTOCOL, LOCALHOST } from '../../constants/product.js';

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
                    loggerCallback(_loglevel: LogLevel, _message: string, _containsPii: boolean) {
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
            let redirectUri: string | null = null;
            const scopes = request.scopes || ["User.Read"];
            let timeoutId: NodeJS.Timeout | null = null;

            const server = http.createServer(async (req, res) => {
                const rawUrl = req.url;
                if (!rawUrl) {
                    res.end('Error occurred loading redirectUrl');
                    reject(new Error("Unable to load redirect URL."));
                    server.close();
                    return;
                }

                const base = redirectUri ?? `${HTTP_PROTOCOL}${LOCALHOST}`;
                const url = new URL(rawUrl, base);
                if (url.pathname !== "/redirect" && url.pathname !== "/") {
                    res.writeHead(404);
                    res.end();
                    return;
                }

                const code = url.searchParams.get('code');
                const error = url.searchParams.get('error');

                if (code) {
                    res.writeHead(200, { 'Content-Type': 'text/html' });
                    res.end('<h1>Authentication successful! You can close this window.</h1>');
                    if (timeoutId) clearTimeout(timeoutId);
                    server.close();

                    try {
                        const result = await this.client.acquireTokenByCode({
                            code,
                            scopes,
                            redirectUri: redirectUri || base
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
                    return;
                }

                if (error) {
                    res.end(`Error occurred: ${error}`);
                    if (timeoutId) clearTimeout(timeoutId);
                    server.close();
                    reject(new Error(error));
                    return;
                }

                res.writeHead(400);
                res.end("Authorization code not found");
            });

            server.on("error", (err) => {
                if (timeoutId) clearTimeout(timeoutId);
                reject(err);
            });

            server.listen(0, LOCALHOST, async () => {
                const address = server.address();
                if (!address || typeof address === "string" || !address.port) {
                    reject(new Error("Invalid loopback address type."));
                    server.close();
                    return;
                }

                redirectUri = `${HTTP_PROTOCOL}${LOCALHOST}:${address.port}/redirect`;
                try {
                    const authUrl = await this.client.getAuthCodeUrl({
                        scopes,
                        redirectUri,
                        loginHint: request.loginHint
                    });
                    await open(authUrl);
                } catch (err) {
                    if (timeoutId) clearTimeout(timeoutId);
                    server.close();
                    reject(err);
                }
            });

            // Timeout after 5 minutes
            timeoutId = setTimeout(() => {
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
