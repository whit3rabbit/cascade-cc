/**
 * File: src/services/auth/OAuthService.ts
 * Role: Utilities for handling OAuth flows and managing tokens.
 */

import http from "node:http";
import fs from 'node:fs';
import {
    createLoopbackServerAlreadyExistsError,
    createUnableToLoadRedirectUrlError,
    createNoLoopbackServerExistsError,
    createInvalidLoopbackAddressTypeError
} from "./ApiKeyManager.js";
import { HTTP_PROTOCOL, LOCALHOST, CLIENT_ID, TOKEN_URL, ROLES_URL, API_KEY_URL, CLAUDE_AI_AUTHORIZE_URL, CONSOLE_AUTHORIZE_URL, MANUAL_REDIRECT_URL } from "../../constants/product.js";
import { IDENTITY_ENDPOINT, IMDS_ENDPOINT } from "../../constants/keys.js";
import { AzureArc } from "../../utils/shared/envContext.js";
import RetryStrategy from "../../utils/http/RetryStrategy.js";
import { getStatsigStorage, setStatsigStorage } from "./MsalAuthService.js";
import { KeychainService } from "./KeychainService.js";
import { getProductName } from "../../utils/shared/product.js";

const HIMDS_EXECUTABLE = "/opt/azcmagent/bin/himds";
const HIMDS_ENDPOINT = "http://localhost:40342";

// Constants
const SCOPES_INFERENCE = "inference";
const SCOPES_DEFAULT = ["profile", "email", "openid"];

/**
 * Interface for token exchange response.
 */
export interface TokenExchangeResponse {
    access_token: string;
    refresh_token?: string;
    expires_in: number;
    scope: string;
    account?: OAuthAccount;
    [key: string]: any;
}

/**
 * Interface for refreshed token data.
 */
export interface RefreshedTokenData {
    accessToken: string;
    refreshToken: string;
    expiresAt: number;
    scopes: string[];
    subscriptionType?: string;
    rateLimitTier?: string;
    account?: OAuthAccount;
}

export interface OAuthAccount {
    accountUuid: string;
    emailAddress: string;
    organizationUuid: string;
    displayName?: string;
    hasExtraUsageEnabled?: boolean;
    billingType?: string;
    organizationRole?: string;
    workspaceRole?: string;
    organizationName?: string;
}

/**
 * Service to handle OAuth flow.
 */
export const OAuthService = {

    /**
     * Gets a valid OAuth token, refreshing if necessary.
     * @returns {Promise<string | null>} The valid access token or null.
     */
    async getValidToken(): Promise<string | null> {
        // 1. Check in-memory storage first
        const storage = getStatsigStorage();
        if (storage.accessToken) {
            // In a real implementation, we'd check expiration here
            return storage.accessToken;
        }

        // 2. Try keychain
        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            const dataStr = KeychainService.readToken(serviceName);
            if (dataStr) {
                try {
                    const rawData = JSON.parse(dataStr);
                    const data = rawData.claudeAiOauth || rawData;
                    if (data.accessToken) {
                        // Check if expiration is near
                        const now = Date.now();
                        const fiveMinutes = 5 * 60 * 1000;
                        if (data.expiresAt && now + fiveMinutes < data.expiresAt) {
                            setStatsigStorage({ accessToken: data.accessToken, oauthAccount: data.account || data.oauthAccount });
                            return data.accessToken;
                        }

                        // Try refresh
                        if (data.refreshToken) {
                            const refreshed = await this.refreshToken(data.refreshToken, data.scopes);
                            await this.saveToken(refreshed);
                            return refreshed.accessToken;
                        }
                    }
                } catch (e) {
                    console.error("[OAuthService] Failed to parse keychain token", e);
                }
            }
        }

        return null;
    },

    /**
     * Saves tokens to persistence.
     */
    async saveToken(data: RefreshedTokenData | TokenExchangeResponse): Promise<void> {
        const accessToken = (data as any).accessToken || (data as TokenExchangeResponse).access_token;
        const refreshToken = (data as any).refreshToken || (data as TokenExchangeResponse).refresh_token;
        const expiresAt = (data as any).expiresAt || (Date.now() + (data as TokenExchangeResponse).expires_in * 1000);
        const account = (data as any).account;
        const scopes = (data as any).scopes || (data as TokenExchangeResponse).scope?.split(" ").filter(Boolean) || [];

        setStatsigStorage({ accessToken, oauthAccount: account });

        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            const tokenData = JSON.stringify({
                claudeAiOauth: {
                    accessToken,
                    refreshToken,
                    expiresAt,
                    account,
                    scopes
                }
            });
            KeychainService.saveToken(serviceName, tokenData);
        }
    },

    /**
     * Builds the authorization URL for the OAuth flow.
     */
    buildAuthUrl({
        codeChallenge,
        state,
        port,
        isManual,
        loginWithClaudeAi,
        inferenceOnly,
        orgUUID,
    }: {
        codeChallenge: string;
        state: string;
        port?: number;
        isManual?: boolean;
        loginWithClaudeAi?: boolean;
        inferenceOnly?: boolean;
        orgUUID?: string;
    }): string {
        const baseUrl = loginWithClaudeAi ? CLAUDE_AI_AUTHORIZE_URL : CONSOLE_AUTHORIZE_URL;
        const url = new URL(baseUrl);
        url.searchParams.append("code", "true");
        url.searchParams.append("client_id", CLIENT_ID);
        url.searchParams.append("response_type", "code");
        url.searchParams.append("redirect_uri", isManual ? MANUAL_REDIRECT_URL : `http://localhost:${port}/callback`);
        const scopes = inferenceOnly ? [SCOPES_INFERENCE] : SCOPES_DEFAULT;
        url.searchParams.append("scope", scopes.join(" "));
        url.searchParams.append("code_challenge", codeChallenge);
        url.searchParams.append("code_challenge_method", "S256");
        url.searchParams.append("state", state);
        if (orgUUID) {
            url.searchParams.append("orgUUID", orgUUID);
        }
        return url.toString();
    },

    /**
     * Exchanges an authorization code for tokens.
     */
    async exchangeToken(
        code: string,
        state: string,
        codeVerifier: string,
        port?: number,
        isManual: boolean = false,
        expiresIn?: number
    ): Promise<TokenExchangeResponse> {
        const body: any = {
            grant_type: "authorization_code",
            code: code,
            redirect_uri: isManual ? MANUAL_REDIRECT_URL : `http://localhost:${port}/callback`,
            client_id: CLIENT_ID,
            code_verifier: codeVerifier,
            state: state,
        };

        if (expiresIn !== undefined) {
            body.expires_in = expiresIn;
        }

        const response = await RetryStrategy.post(TOKEN_URL, body, {
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (response.status !== 200) {
            const errorMessage =
                response.status === 401
                    ? "Authentication failed: Invalid authorization code"
                    : `Token exchange failed (${response.status}): ${response.statusText}`;
            throw new Error(errorMessage);
        }

        // track("tengu_oauth_token_exchange_success", {});
        return response.data;
    },

    /**
     * Refreshes an access token using a refresh token.
     */
    async refreshToken(refreshTokenValue: string, scopes?: string[]): Promise<RefreshedTokenData> {
        const body = {
            grant_type: "refresh_token",
            refresh_token: refreshTokenValue,
            client_id: CLIENT_ID,
            scope: (scopes && scopes.length > 0) ? scopes.join(" ") : SCOPES_DEFAULT.join(" "),
        };

        try {
            const response = await RetryStrategy.post(TOKEN_URL, body, {
                headers: {
                    "Content-Type": "application/json",
                },
            });

            if (response.status !== 200) {
                throw new Error(`Token refresh failed: ${response.statusText}`);
            }

            const data = response.data;
            const {
                access_token: accessToken,
                refresh_token: newRefreshToken = refreshTokenValue,
                expires_in: expiresIn,
            } = data;
            const expiresAt = Date.now() + expiresIn * 1000;
            const scopes = data.scope?.split(" ").filter(Boolean) ?? [];

            // track("tengu_oauth_token_refresh_success", {});

            return {
                accessToken,
                refreshToken: newRefreshToken,
                expiresAt,
                scopes,
                // These would come from a profile fetch typically
                subscriptionType: undefined,
                rateLimitTier: undefined,
            };
        } catch (error) {
            // track("tengu_oauth_token_refresh_failure", { error: error.message });
            throw error;
        }
    },

    /**
     * Fetches the user profile.
     */
    async fetchProfile(accessToken: string): Promise<any> {
        const response = await RetryStrategy.get(`${CONSOLE_AUTHORIZE_URL.replace("/oauth/authorize", "")}/api/oauth/profile`, {
            headers: {
                Authorization: `Bearer ${accessToken}`,
                "Content-Type": "application/json"
            }
        });

        if (response.status !== 200) {
            throw new Error(`Failed to fetch profile: ${response.statusText}`);
        }

        return response.data;
    },

    /**
     * Fetches user roles.
     */
    async fetchRoles(accessToken: string): Promise<any> {
        const response = await RetryStrategy.get(ROLES_URL, {
            headers: {
                Authorization: `Bearer ${accessToken}`,
                "Content-Type": "application/json"
            }
        });

        if (response.status !== 200) {
            throw new Error(`Failed to fetch roles: ${response.statusText}`);
        }

        return response.data;
    },

    /**
     * Logs out the user by clearing local persistence.
     */
    async logout(): Promise<void> {
        setStatsigStorage({ accessToken: undefined, oauthAccount: undefined });
        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            KeychainService.deleteToken(serviceName);
        }
    }
};

/**
 * Handles the local loopback server for creating an auth callback.
 */
export class LoopbackServerHandler {
    private server: http.Server | undefined;

    /**
     * Starts the loopback server on a random port and returns it.
     */
    async start(): Promise<number> {
        if (this.server) {
            throw createLoopbackServerAlreadyExistsError();
        }

        return new Promise((resolve, reject) => {
            this.server = http.createServer();
            this.server.on('error', reject);
            this.server.listen(0, "127.0.0.1", () => {
                const address = this.server!.address();
                if (typeof address === 'string' || !address?.port) {
                    this.closeServer();
                    reject(createInvalidLoopbackAddressTypeError());
                    return;
                }
                resolve(address.port);
            });
        });
    }

    /**
     * Listens for the auth code on the current server.
     */
    async listenForAuthCode(successMessage?: string, errorMessage?: string): Promise<Record<string, string>> {
        if (!this.server) {
            await this.start();
        }

        return new Promise((resolve, reject) => {
            this.server!.on('request', (req, res) => {
                const reqUrl = req.url;
                if (!reqUrl) {
                    res.end(errorMessage || "Error occurred loading redirectUrl");
                    reject(createUnableToLoadRedirectUrlError());
                    return;
                }

                if (reqUrl === "/") {
                    res.end(successMessage || "Auth code was successfully acquired. You can close this window now.");
                    return;
                }

                try {
                    const redirectUri = this.getRedirectUri();
                    const urlObj = new URL(reqUrl, redirectUri);
                    const params = Object.fromEntries(urlObj.searchParams);

                    if (params.code) {
                        res.writeHead(302, { location: redirectUri });
                        res.end();
                        resolve(params);
                    } else if (params.error) {
                        res.end(errorMessage || `Error occurred: ${params.error}`);
                        reject(new Error(params.error));
                    }
                } catch (e) {
                    res.end(errorMessage || String(e));
                    reject(e);
                }
            });
        });
    }

    getRedirectUri(): string {
        if (!this.server || !this.server.listening) {
            throw createNoLoopbackServerExistsError();
        }

        const serverAddress = this.server.address();
        if (typeof serverAddress === "string" || !serverAddress?.port) {
            this.closeServer();
            throw createInvalidLoopbackAddressTypeError();
        }

        const port = serverAddress.port;
        return `${HTTP_PROTOCOL}${LOCALHOST}:${port}`;
    }

    closeServer(): void {
        if (this.server) {
            this.server.close();
            if (typeof (this.server as any).closeAllConnections === "function") {
                (this.server as any).closeAllConnections();
            }
            this.server.unref();
            this.server = undefined;
        }
    }
}
