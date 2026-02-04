/**
 * File: src/services/auth/OAuthService.ts
 * Role: Utilities for handling OAuth flows and managing tokens, aligned with 2.1.19.
 */

import * as http from "node:http";
import { createHash, randomBytes } from "node:crypto";
import {
    CLIENT_ID,
    TOKEN_URL,
    ROLES_URL,
    CLAUDE_AI_AUTHORIZE_URL,
    CONSOLE_AUTHORIZE_URL,
    MANUAL_REDIRECT_URL,
    OAUTH_BETA_HEADER,
    PROFILE_URL
} from "../../constants/product.js";
import { KeychainService } from "./KeychainService.js";
import { getProductName } from "../../utils/shared/product.js";
import RetryStrategy from "../../utils/http/RetryStrategy.js";

// Helper for base64url encoding (chunk755: bD6 logic)
function base64url(buffer: Buffer): string {
    return buffer.toString("base64")
        .replace(/\+/g, "-")
        .replace(/\//g, "_")
        .replace(/=/g, "");
}

// PKCE Helpers (chunk755: mZ7, gZ7 logic)
export function createCodeVerifier(): string {
    return base64url(randomBytes(32));
}

export function createCodeChallenge(verifier: string): string {
    const hash = createHash("sha256").update(verifier).digest();
    return base64url(hash);
}

export function createOAuthState(): string {
    return base64url(randomBytes(32));
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
    subscriptionType?: string | null;
    rateLimitTier?: string | null;
}

export interface TokenExchangeResponse {
    access_token: string;
    refresh_token: string;
    expires_in: number;
    scope: string;
    account?: {
        uuid: string;
        email_address: string;
    };
    organization?: {
        uuid: string;
    };
}

export interface AuthSession {
    accessToken: string;
    refreshToken: string;
    expiresAt: number;
    scopes: string[];
    account?: OAuthAccount;
}

const SCOPES_DEFAULT = ["profile", "email", "openid"];
const SCOPES_INFERENCE = "inference";

let currentSession: AuthSession | null = null;

/**
 * Service to handle OAuth flow.
 */
export const OAuthService = {
    /**
     * Gets a valid OAuth token, refreshing if necessary.
     */
    async getValidToken(): Promise<string | null> {
        if (currentSession?.accessToken && currentSession.expiresAt > Date.now() + 60000) {
            return currentSession.accessToken;
        }

        const persisted = await this.loadSession();
        if (persisted) {
            currentSession = persisted;
            if (currentSession.expiresAt > Date.now() + 60000) {
                return currentSession.accessToken;
            }
            // Try refresh
            try {
                const refreshed = await this.refreshToken(currentSession.refreshToken, currentSession.scopes);
                await this.saveSession(refreshed);
                return refreshed.accessToken;
            } catch (e) {
                console.error("[OAuth] Refresh failed:", e);
                await this.logout();
            }
        }
        return null;
    },

    /**
     * Builds the authorization URL (chunk809: KK6 logic).
     */
    buildAuthUrl(options: {
        codeChallenge: string;
        state: string;
        port: number;
        isManual?: boolean;
        loginWithClaudeAi?: boolean;
        inferenceOnly?: boolean;
        orgUUID?: string;
    }): string {
        const baseUrl = options.loginWithClaudeAi ? CLAUDE_AI_AUTHORIZE_URL : CONSOLE_AUTHORIZE_URL;
        const url = new URL(baseUrl);
        url.searchParams.append("response_type", "code");
        url.searchParams.append("client_id", CLIENT_ID);
        url.searchParams.append("redirect_uri", options.isManual ? MANUAL_REDIRECT_URL : `http://localhost:${options.port}/callback`);
        url.searchParams.append("code_challenge", options.codeChallenge);
        url.searchParams.append("code_challenge_method", "S256");
        url.searchParams.append("state", options.state);

        const scopes = options.inferenceOnly ? [SCOPES_INFERENCE] : SCOPES_DEFAULT;
        url.searchParams.append("scope", scopes.join(" "));

        if (options.orgUUID) {
            url.searchParams.append("org_uuid", options.orgUUID);
        }
        return url.toString();
    },

    /**
     * Maps organization type to subscription type (chunk407: YK6 logic).
     */
    mapSubscriptionType(orgType?: string | null): string | null {
        switch (orgType) {
            case "claude_max": return "max";
            case "claude_pro": return "pro";
            case "claude_enterprise": return "enterprise";
            case "claude_team": return "team";
            default: return null;
        }
    },

    /**
     * Exchanges code for tokens (chunk809: TN4 logic).
     */
    async exchangeToken(
        code: string,
        state: string,
        codeVerifier: string,
        port: number,
        isManual: boolean = false,
        expiresIn?: number
    ): Promise<TokenExchangeResponse> {
        const body: any = {
            grant_type: "authorization_code",
            code,
            client_id: CLIENT_ID,
            code_verifier: codeVerifier,
            redirect_uri: isManual ? MANUAL_REDIRECT_URL : `http://localhost:${port}/callback`,
            state
        };

        if (expiresIn) body.expires_in = expiresIn;

        const response = await RetryStrategy.post(TOKEN_URL, body, {
            headers: { "Content-Type": "application/json" }
        });

        if (response.status !== 200) {
            throw new Error(`Token exchange failed: ${response.statusText}`);
        }

        return response.data;
    },

    /**
     * Refreshes access token.
     */
    async refreshToken(refreshToken: string, scopes: string[]): Promise<AuthSession> {
        const body = {
            grant_type: "refresh_token",
            refresh_token: refreshToken,
            client_id: CLIENT_ID,
            scope: scopes.join(" ")
        };

        const response = await RetryStrategy.post(TOKEN_URL, body, {
            headers: { "Content-Type": "application/json" }
        });

        if (response.status !== 200) {
            throw new Error(`Token refresh failed: ${response.statusText}`);
        }

        const data = response.data;
        const accessToken = data.access_token;
        const newRefreshToken = data.refresh_token || refreshToken;

        // Fetch profile and roles (aligned with chunk809 and chunk407 YK6)
        const profile = await this.fetchProfile(accessToken);
        const roles = await this.fetchRoles(accessToken);

        // Map organization_type to subscriptionType (chunk407: YK6 logic)
        const subscriptionType = this.mapSubscriptionType(profile.organization?.organization_type);

        return {
            accessToken,
            refreshToken: newRefreshToken,
            expiresAt: Date.now() + (data.expires_in * 1000),
            scopes: data.scope?.split(" ") || scopes,
            account: {
                accountUuid: profile.account.uuid,
                emailAddress: profile.account.email,
                organizationUuid: profile.organization.uuid,
                displayName: profile.account.display_name,
                hasExtraUsageEnabled: profile.organization.has_extra_usage_enabled,
                billingType: profile.organization.billing_type,
                organizationRole: roles.organization_role,
                workspaceRole: roles.workspace_role,
                organizationName: roles.organization_name,
                subscriptionType,
                rateLimitTier: profile.organization?.rate_limit_tier ?? null
            }
        };
    },

    /**
     * Persists session to keychain.
     */
    async saveSession(session: AuthSession): Promise<void> {
        currentSession = session;
        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            KeychainService.saveToken(serviceName, JSON.stringify(session));
        }
    },

    /**
     * Loads session from keychain.
     */
    async loadSession(): Promise<AuthSession | null> {
        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            const data = KeychainService.readToken(serviceName);
            if (data) {
                try {
                    return JSON.parse(data);
                } catch {
                    return null;
                }
            }
        }
        return null;
    },

    /**
     * Logs out (chunk808: yN6 cleanup).
     */
    async logout(): Promise<void> {
        currentSession = null;
        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            KeychainService.deleteToken(serviceName);
        }
        // Additional cleanup like clearing remote-settings cache would go here
    },

    /**
     * Orchestrates the full login flow (chunk809: startOAuthFlow logic).
     */
    async login(options: {
        onUrl: (url: string) => Promise<void>,
        orgUUID?: string,
        loginWithClaudeAi?: boolean
    }): Promise<AuthSession> {
        const server = new LocalAuthServer();
        const port = await server.start();
        const state = createOAuthState();
        const verifier = createCodeVerifier();
        const challenge = createCodeChallenge(verifier);

        const url = this.buildAuthUrl({
            codeChallenge: challenge,
            state,
            port,
            orgUUID: options.orgUUID,
            loginWithClaudeAi: options.loginWithClaudeAi
        });

        try {
            const code = await server.waitForCode(state, () => options.onUrl(url));

            const tokenResponse = await this.exchangeToken(code, state, verifier, port);

            // Fetch profile and roles (chunk809: YK6 and zK6 logic)
            const profile = await this.fetchProfile(tokenResponse.access_token);
            const roles = await this.fetchRoles(tokenResponse.access_token);

            // Map organization_type to subscriptionType (chunk407: YK6 logic)
            const subscriptionType = this.mapSubscriptionType(profile.organization?.organization_type);

            const session: AuthSession = {
                accessToken: tokenResponse.access_token,
                refreshToken: tokenResponse.refresh_token,
                expiresAt: Date.now() + (tokenResponse.expires_in * 1000),
                scopes: tokenResponse.scope.split(" "),
                account: {
                    accountUuid: profile.account.uuid,
                    emailAddress: profile.account.email,
                    organizationUuid: profile.organization.uuid,
                    displayName: profile.account.display_name,
                    hasExtraUsageEnabled: profile.organization.has_extra_usage_enabled,
                    billingType: profile.organization.billing_type,
                    organizationRole: roles.organization_role,
                    workspaceRole: roles.workspace_role,
                    organizationName: roles.organization_name,
                    subscriptionType,
                    rateLimitTier: profile.organization?.rate_limit_tier ?? null
                }
            };

            await this.saveSession(session);
            return session;
        } finally {
            server.close();
        }
    },

    /**
     * Fetches the user profile (chunk809: YK6 logic).
     */
    async fetchProfile(accessToken: string): Promise<any> {
        const response = await RetryStrategy.get(PROFILE_URL, {
            headers: {
                Authorization: `Bearer ${accessToken}`,
                "anthropic-beta": OAUTH_BETA_HEADER
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
                "anthropic-beta": OAUTH_BETA_HEADER
            }
        });
        if (response.status !== 200) {
            throw new Error(`Failed to fetch roles: ${response.statusText}`);
        }
        return response.data;
    }
};

/**
 * Port of chunk754 hD6 (Local Auth Server).
 */
export class LocalAuthServer {
    private server: http.Server;
    private port: number = 0;
    private expectedState: string | null = null;
    private resolve: ((code: string) => void) | null = null;
    private reject: ((err: Error) => void) | null = null;

    constructor() {
        this.server = http.createServer();
    }

    async start(): Promise<number> {
        return new Promise((resolve, reject) => {
            this.server.on("error", reject);
            this.server.listen(0, "127.0.0.1", () => {
                const addr = this.server.address();
                this.port = (addr as any).port;
                resolve(this.port);
            });
        });
    }

    async waitForCode(state: string, onStart: () => void): Promise<string> {
        this.expectedState = state;
        return new Promise((resolve, reject) => {
            this.resolve = resolve;
            this.reject = reject;
            this.server.on("request", (req, res) => {
                const url = new URL(req.url || "", `http://localhost:${this.port}`);
                if (url.pathname !== "/callback") {
                    res.writeHead(404);
                    res.end();
                    return;
                }

                const code = url.searchParams.get("code");
                const returnedState = url.searchParams.get("state");

                if (!code) {
                    res.writeHead(400);
                    res.end("Missing code");
                    reject(new Error("No code received"));
                    return;
                }

                if (returnedState !== this.expectedState) {
                    res.writeHead(400);
                    res.end("Invalid state");
                    reject(new Error("State mismatch"));
                    return;
                }

                // Chunk754 style success redirect
                res.writeHead(302, { Location: "https://claude.ai/login_successful" });
                res.end();

                resolve(code);
                this.close();
            });
            onStart();
        });
    }

    close() {
        this.server.close();
        this.server.unref();
    }
}
