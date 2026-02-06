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
    CONSOLE_SUCCESS_URL,
    CLAUDEAI_SUCCESS_URL,
    MANUAL_REDIRECT_URL,
    OAUTH_BETA_HEADER,
    PROFILE_URL
} from "../../constants/product.js";
import {
    OAUTH_SCOPE_INFERENCE,
    OAUTH_SCOPES_ALL,
    OAUTH_SCOPES_DEFAULT,
    hasInferenceScope,
    parseOAuthScopes
} from "../../constants/oauth.js";
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
    subscriptionType?: string | null;
    rateLimitTier?: string | null;
}

export interface OAuthFlowOptions {
    orgUUID?: string;
    loginWithClaudeAi?: boolean;
    inferenceOnly?: boolean;
    expiresIn?: number;
}

let currentSession: AuthSession | null = null;

/**
 * Service to handle OAuth flow.
 */
export const OAuthService = {
    /**
     * Gets a valid OAuth token, refreshing if necessary.
     */
    async getValidToken(forceRefresh: boolean = false): Promise<string | null> {
        if (!forceRefresh && currentSession?.accessToken && currentSession.expiresAt > Date.now() + 60000) {
            return currentSession.accessToken;
        }

        const persisted = await this.loadSession();
        if (persisted) {
            currentSession = persisted;
            if (!forceRefresh && currentSession.expiresAt > Date.now() + 60000) {
                return currentSession.accessToken;
            }
            // Try refresh
            try {
                const refreshed = await this.refreshTokenLocked(currentSession.refreshToken, currentSession.scopes);
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
     * Builds the OAuth authorization URL (chunk406: clearConversation_3 logic).
     */
    buildAuthUrl(options: {
        codeChallenge: string,
        state: string,
        port: number,
        isManual?: boolean,
        loginWithClaudeAi?: boolean,
        inferenceOnly?: boolean,
        orgUUID?: string
    }): string {
        const baseUrl = options.loginWithClaudeAi ? CLAUDE_AI_AUTHORIZE_URL : CONSOLE_AUTHORIZE_URL;
        const url = new URL(baseUrl);

        url.searchParams.append("code", "true");
        url.searchParams.append("client_id", CLIENT_ID);
        url.searchParams.append("response_type", "code");
        url.searchParams.append("redirect_uri", options.isManual ? MANUAL_REDIRECT_URL : `http://localhost:${options.port}/callback`);

        // Scope selection (chunk406: clearConversation_3 logic)
        // o58 = ["org:create_api_key", "user:profile", "user:inference", "user:sessions:claude_code", "user:mcp_servers"]
        const scopes = options.inferenceOnly ? [OAUTH_SCOPE_INFERENCE] : OAUTH_SCOPES_ALL;
        url.searchParams.append("scope", scopes.join(" "));

        url.searchParams.append("code_challenge", options.codeChallenge);
        url.searchParams.append("code_challenge_method", "S256");
        url.searchParams.append("state", options.state);

        if (options.orgUUID) {
            url.searchParams.append("orgUUID", options.orgUUID);
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
            throw new Error(response.status === 401
                ? "Authentication failed: Invalid authorization code"
                : `Token exchange failed (${response.status}): ${response.statusText}`);
        }

        return response.data;
    },

    /**
     * Refreshes access token.
     */
    /**
     * Refreshes access token with cross-process locking (aligned with chunk1471: clearConversation_17).
     */
    async refreshTokenLocked(refreshToken: string, scopes: string[], retryCount: number = 0): Promise<AuthSession> {
        const { getClaudePaths } = await import("../../utils/shared/runtimeAndEnv.js");
        const { tryAcquireLock } = await import("../../utils/process/ProcessLock.js");
        const { mkdirSync } = await import("node:fs");

        const lockDir = getClaudePaths().locks;
        try {
            mkdirSync(lockDir, { recursive: true });
        } catch { }

        const release = await tryAcquireLock("oauth_refresh", lockDir);
        if (!release) {
            if (retryCount < 5) {
                // Wait 1-2 seconds and retry (chunk1471 logic)
                await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));

                // Re-check session before retrying (chunk1471: FK.cache.clear followed by FK())
                const persisted = await this.loadSession();
                if (persisted && persisted.refreshToken !== refreshToken) {
                    // Someone else refreshed it!
                    return persisted;
                }

                return this.refreshTokenLocked(refreshToken, scopes, retryCount + 1);
            }
            throw new Error("Failed to acquire lock for token refresh after multiple attempts.");
        }

        try {
            // Re-check after acquiring lock (chunk1471 logic)
            const persisted = await this.loadSession();
            if (persisted && persisted.refreshToken !== refreshToken && persisted.expiresAt > Date.now() + 60000) {
                return persisted;
            }

            return await this.refreshToken(refreshToken, scopes);
        } finally {
            release();
        }
    },

    /**
     * Refreshes access token (internal).
     */
    async refreshToken(refreshToken: string, scopes: string[]): Promise<AuthSession> {
        const scopeList = scopes.length ? scopes : OAUTH_SCOPES_DEFAULT;
        const body = {
            grant_type: "refresh_token",
            refresh_token: refreshToken,
            client_id: CLIENT_ID,
            scope: scopeList.join(" ")
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

        const parsedScopes = parseOAuthScopes(data.scope);
        const finalScopes = parsedScopes.length ? parsedScopes : scopeList;

        return {
            accessToken,
            refreshToken: newRefreshToken,
            expiresAt: Date.now() + (data.expires_in * 1000),
            scopes: finalScopes,
            subscriptionType,
            rateLimitTier: profile.organization?.rate_limit_tier ?? null,
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
    async logout(options: { clearOnboarding?: boolean } = { clearOnboarding: true }): Promise<void> {
        currentSession = null;
        if (KeychainService.isAvailable()) {
            const serviceName = getProductName("-credentials");
            KeychainService.deleteToken(serviceName);
        }
        // Additional cleanup: clearing remote-settings cache and resetting user state (aligned with chunk808: yN6)
        const { clearRemoteSettingsCache } = await import("../config/RemoteSettingsService.js");
        const { updateSettings } = await import("../config/SettingsService.js");
        const AuthService = await import("./AuthService.js");
        const StatsigManager = await import("../statsig/StatsigManager.js");
        const { setStatsigStorage } = await import("./MsalAuthService.js");

        clearRemoteSettingsCache();
        if (options.clearOnboarding) {
            updateSettings({
                onboardingComplete: false,
                subscriptionNoticeCount: 0,
                hasAvailableSubscription: false
            });
        }
        AuthService.logout();
        StatsigManager.logout();
        setStatsigStorage({ oauthAccount: null, accessToken: null });
    },

    /**
     * Orchestrates the full login flow (chunk809: startOAuthFlow logic).
     */
    async startOAuthFlow(options: OAuthFlowOptions & {
        onUrl: (manualUrl: string, autoUrl: string) => Promise<void>
    }): Promise<AuthSession> {
        const flow = new OAuthFlow();
        try {
            return await flow.startOAuthFlow(options.onUrl, options);
        } finally {
            flow.cleanup();
        }
    },

    /**
     * Backwards-compatible alias for startOAuthFlow.
     */
    async login(options: OAuthFlowOptions & {
        onUrl: (manualUrl: string, autoUrl: string) => Promise<void>
    }): Promise<AuthSession> {
        return this.startOAuthFlow(options);
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

export class OAuthFlow {
    private codeVerifier: string;
    private authCodeListener: LocalAuthServer | null = null;
    private port: number | null = null;
    private manualAuthCodeResolver: ((code: string) => void) | null = null;

    constructor() {
        this.codeVerifier = createCodeVerifier();
    }

    async startOAuthFlow(
        onUrl: (manualUrl: string, autoUrl: string) => Promise<void>,
        options: OAuthFlowOptions = {}
    ): Promise<AuthSession> {
        this.authCodeListener = new LocalAuthServer();
        this.port = await this.authCodeListener.start();
        const port = this.port;
        if (port === null) {
            throw new Error("OAuth callback server did not return a port.");
        }

        const state = createOAuthState();
        const challenge = createCodeChallenge(this.codeVerifier);

        const common = {
            codeChallenge: challenge,
            state,
            port,
            loginWithClaudeAi: options.loginWithClaudeAi,
            inferenceOnly: options.inferenceOnly,
            orgUUID: options.orgUUID
        };

        const manualUrl = OAuthService.buildAuthUrl({ ...common, isManual: true });
        const autoUrl = OAuthService.buildAuthUrl({ ...common, isManual: false });

        try {
            const code = await this.waitForAuthorizationCode(state, async () => onUrl(manualUrl, autoUrl));
            const automatic = this.authCodeListener?.hasPendingResponse() ?? false;

            const tokenResponse = await OAuthService.exchangeToken(
                code,
                state,
                this.codeVerifier,
                port,
                !automatic,
                options.expiresIn
            );

            await OAuthService.logout({ clearOnboarding: false });

            const profile = await OAuthService.fetchProfile(tokenResponse.access_token);
            const roles = await OAuthService.fetchRoles(tokenResponse.access_token);
            const subscriptionType = OAuthService.mapSubscriptionType(profile.organization?.organization_type);

            const parsedScopes = parseOAuthScopes(tokenResponse.scope);
            const scopes = parsedScopes.length
                ? parsedScopes
                : options.inferenceOnly
                    ? [OAUTH_SCOPE_INFERENCE]
                    : OAUTH_SCOPES_DEFAULT;

            const session: AuthSession = {
                accessToken: tokenResponse.access_token,
                refreshToken: tokenResponse.refresh_token,
                expiresAt: Date.now() + (tokenResponse.expires_in * 1000),
                scopes,
                subscriptionType,
                rateLimitTier: profile.organization?.rate_limit_tier ?? null,
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

            if (automatic) {
                this.authCodeListener?.handleSuccessRedirect(session.scopes);
            }

            const { setStatsigStorage } = await import("./MsalAuthService.js");
            setStatsigStorage({ oauthAccount: session.account ?? null, accessToken: session.accessToken });

            await OAuthService.saveSession(session);
            return session;
        } catch (e) {
            if (this.authCodeListener?.hasPendingResponse()) {
                this.authCodeListener.handleErrorRedirect();
            }
            throw e;
        } finally {
            this.authCodeListener?.close();
        }
    }

    handleManualAuthCodeInput(input: { authorizationCode: string; state?: string }) {
        if (this.manualAuthCodeResolver) {
            this.manualAuthCodeResolver(input.authorizationCode);
            this.manualAuthCodeResolver = null;
            this.authCodeListener?.close();
        }
    }

    cleanup() {
        this.authCodeListener?.close();
        this.manualAuthCodeResolver = null;
    }

    private async waitForAuthorizationCode(state: string, onStart: () => void): Promise<string> {
        return new Promise((resolve, reject) => {
            this.manualAuthCodeResolver = resolve;
            const listener = this.authCodeListener;
            if (!listener) {
                reject(new Error("OAuth callback server is not initialized."));
                return;
            }
            listener.waitForAuthorization(state, onStart).then(code => {
                this.manualAuthCodeResolver = null;
                resolve(code);
            }).catch(err => {
                this.manualAuthCodeResolver = null;
                reject(err);
            });
        });
    }
}

/**
 * Port of chunk754 hD6 (Local Auth Server).
 */
export class LocalAuthServer {
    private localServer: http.Server;
    private port = 0;
    private promiseResolver: ((code: string) => void) | null = null;
    private promiseRejecter: ((err: Error) => void) | null = null;
    private expectedState: string | null = null;
    private pendingResponse: http.ServerResponse | null = null;
    private callbackPath: string;

    constructor(callbackPath: string = "/callback") {
        this.localServer = http.createServer();
        this.callbackPath = callbackPath;
    }

    async start(port?: number): Promise<number> {
        return new Promise((resolve, reject) => {
            this.localServer.once("error", err => {
                reject(new Error(`Failed to start OAuth callback server: ${err.message}`));
            });
            this.localServer.listen(port ?? 0, "localhost", () => {
                const addr = this.localServer.address();
                this.port = (addr as any).port;
                resolve(this.port);
            });
        });
    }

    getPort(): number {
        return this.port;
    }

    hasPendingResponse(): boolean {
        return this.pendingResponse !== null;
    }

    async waitForAuthorization(state: string, onStart: () => void): Promise<string> {
        return new Promise((resolve, reject) => {
            this.promiseResolver = resolve;
            this.promiseRejecter = reject;
            this.expectedState = state;
            this.startLocalListener(onStart);
        });
    }

    handleSuccessRedirect(scopes: string[], handler?: (res: http.ServerResponse, scopes: string[]) => void) {
        if (!this.pendingResponse) {
            return;
        }
        if (handler) {
            handler(this.pendingResponse, scopes);
            this.pendingResponse = null;
            return;
        }

        const redirectUrl = hasInferenceScope(scopes) ? CLAUDEAI_SUCCESS_URL : CONSOLE_SUCCESS_URL;
        this.pendingResponse.writeHead(302, { Location: redirectUrl });
        this.pendingResponse.end();
        this.pendingResponse = null;
    }

    handleErrorRedirect() {
        if (!this.pendingResponse) {
            return;
        }

        this.pendingResponse.writeHead(302, { Location: CLAUDEAI_SUCCESS_URL });
        this.pendingResponse.end();
        this.pendingResponse = null;
    }

    private startLocalListener(onStart: () => void) {
        this.localServer.on("request", this.handleRedirect.bind(this));
        this.localServer.on("error", this.handleError.bind(this));
        onStart();
    }

    private handleRedirect(req: http.IncomingMessage, res: http.ServerResponse) {
        const url = new URL(req.url || "", `http://${req.headers.host || "localhost"}`);
        if (url.pathname !== this.callbackPath) {
            res.writeHead(404);
            res.end();
            return;
        }

        const code = url.searchParams.get("code") ?? undefined;
        const state = url.searchParams.get("state") ?? undefined;
        this.validateAndRespond(code, state, res);
    }

    private validateAndRespond(code: string | undefined, state: string | undefined, res: http.ServerResponse) {
        if (!code) {
            res.writeHead(400);
            res.end("Authorization code not found");
            this.reject(new Error("No authorization code received"));
            return;
        }
        if (state !== this.expectedState) {
            res.writeHead(400);
            res.end("Invalid state parameter");
            this.reject(new Error("Invalid state parameter"));
            return;
        }
        this.pendingResponse = res;
        this.resolve(code);
    }

    private handleError(err: Error) {
        console.error("[OAuth] Callback server error:", err);
        this.close();
        this.reject(err);
    }

    private resolve(code: string) {
        if (this.promiseResolver) {
            this.promiseResolver(code);
            this.promiseResolver = null;
            this.promiseRejecter = null;
        }
    }

    private reject(err: Error) {
        if (this.promiseRejecter) {
            this.promiseRejecter(err);
            this.promiseResolver = null;
            this.promiseRejecter = null;
        }
    }

    close() {
        if (this.pendingResponse) {
            this.handleErrorRedirect();
        }
        if (this.localServer) {
            this.localServer.removeAllListeners();
            this.localServer.close();
        }
    }
}
