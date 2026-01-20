import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { URL } from "node:url";
import { randomBytes } from "node:crypto";
import { log } from "../logger/loggerService.js";
import { getSettings, updateSettings } from "../terminal/settings.js";
import axios from "axios";

const logger = log("mcp-oauth");

export interface OAuthMetadata {
    authorization_endpoint: string;
    token_endpoint: string;
    registration_endpoint?: string;
    scopes_supported?: string[];
    default_scope?: string;
    scope?: string;
    revocation_endpoint?: string;
}

export interface OAuthTokenResponse {
    access_token: string;
    refresh_token?: string;
    expires_in: number;
    scope?: string;
    token_type: string;
}

export class McpOAuthController {
    private _state?: string;
    private _codeVerifier?: string;
    private _authorizationUrl?: string;
    private _metadata?: OAuthMetadata;

    constructor(
        public readonly serverName: string,
        public readonly serverConfig: any,
        public readonly redirectUri: string
    ) { }

    async getState(): Promise<string> {
        if (!this._state) {
            this._state = randomBytes(32).toString('base64url');
        }
        return this._state;
    }

    async getAuthorizationUrl(metadata: OAuthMetadata): Promise<string> {
        this._metadata = metadata;
        const state = await this.getState();
        this._codeVerifier = randomBytes(32).toString('base64url');

        const url = new URL(metadata.authorization_endpoint);
        url.searchParams.set("response_type", "code");
        url.searchParams.set("client_id", this.getClientId());
        url.searchParams.set("redirect_uri", this.redirectUri);
        url.searchParams.set("state", state);

        const scope = metadata.scope || metadata.default_scope || metadata.scopes_supported?.join(" ");
        if (scope) url.searchParams.set("scope", scope);

        this._authorizationUrl = url.toString();
        return this._authorizationUrl;
    }

    private getClientId(): string {
        // In real implementation, this would be fetched or registered
        return this.serverConfig.clientId || "claude-code";
    }

    async saveTokens(tokens: OAuthTokenResponse) {
        const settings = getSettings("userSettings");
        const mcpOAuth = { ...(settings.mcpOAuth || {}) };
        const key = this.getStorageKey();

        mcpOAuth[key] = {
            serverName: this.serverName,
            serverUrl: this.serverConfig.url,
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token,
            expiresAt: Date.now() + (tokens.expires_in * 1000),
            scope: tokens.scope
        };

        updateSettings("userSettings", { mcpOAuth });
    }

    async getTokens(): Promise<OAuthTokenResponse | null> {
        const settings = getSettings("userSettings");
        const key = this.getStorageKey();
        const data = settings.mcpOAuth?.[key];

        if (!data) return null;

        const expiresIn = (data.expiresAt - Date.now()) / 1000;
        if (expiresIn <= 0 && !data.refreshToken) return null;

        if (expiresIn <= 300 && data.refreshToken) {
            return await this.refreshTokens(data.refreshToken);
        }

        return {
            access_token: data.accessToken,
            refresh_token: data.refreshToken,
            expires_in: expiresIn,
            scope: data.scope,
            token_type: "Bearer"
        };
    }

    private getStorageKey(): string {
        return `${this.serverName}:${this.serverConfig.url}`;
    }

    async refreshTokens(refreshToken: string): Promise<OAuthTokenResponse | null> {
        logger.info(`Refreshing OAuth tokens for ${this.serverName}`);
        // Implementation would use this._metadata.token_endpoint
        return null; // Mocked
    }

    async exchangeCode(code: string): Promise<OAuthTokenResponse> {
        if (!this._metadata || !this._codeVerifier) {
            throw new Error("OAuth flow not initialized");
        }

        const params = new URLSearchParams();
        params.set("grant_type", "authorization_code");
        params.set("code", code);
        params.set("redirect_uri", this.redirectUri);
        params.set("client_id", this.getClientId());
        params.set("code_verifier", this._codeVerifier);

        const response = await axios.post(this._metadata.token_endpoint, params, {
            headers: { "Content-Type": "application/x-www-form-urlencoded" }
        });

        return response.data;
    }

    async revokeTokens(): Promise<void> {
        const settings = getSettings("userSettings");
        const key = this.getStorageKey();
        const data = settings.mcpOAuth?.[key];

        if (!data) return;

        // Ensure we have metadata
        if (!this._metadata) {
            try {
                const metadataUrl = new URL(this.serverConfig.url);
                metadataUrl.pathname = "/.well-known/mcp-oauth-metadata";
                this._metadata = (await axios.get(metadataUrl.toString())).data;
            } catch (e) {
                logger.warn(`Could not fetch metadata for revocation: ${e}`);
                return;
            }
        }

        if (this._metadata?.revocation_endpoint) {
            const revoke = async (token: string, type: string) => {
                const params = new URLSearchParams();
                params.set("token", token);
                params.set("token_type_hint", type);
                params.set("client_id", this.getClientId());

                try {
                    await axios.post(this._metadata!.revocation_endpoint!, params, {
                        headers: { "Content-Type": "application/x-www-form-urlencoded" }
                    });
                } catch (e) {
                    logger.warn(`Failed to revoke ${type}: ${e}`);
                }
            };

            if (data.refreshToken) await revoke(data.refreshToken, "refresh_token");
            if (data.accessToken) await revoke(data.accessToken, "access_token");
        }

        // Clear settings
        const mcpOAuth = { ...(settings.mcpOAuth || {}) };
        delete mcpOAuth[key];
        updateSettings("userSettings", { mcpOAuth });
    }
}

async function findAvailablePort(): Promise<number> {
    const { createServer } = await import("node:http");
    const min = 3000;
    const max = 3999;

    // Try environment variable first
    const envPort = parseInt(process.env.MCP_OAUTH_CALLBACK_PORT || "", 10);
    if (envPort > 0) return envPort;

    // Try random ports
    for (let i = 0; i < 100; i++) {
        const port = Math.floor(Math.random() * (max - min + 1)) + min;
        try {
            await new Promise<void>((resolve, reject) => {
                const server = createServer();
                server.once('error', reject);
                server.listen(port, () => {
                    server.close(() => resolve());
                });
            });
            return port;
        } catch { }
    }
    return 3118; // Fallback
}


export async function startMcpOAuthFlow(serverName: string, config: any, onOpenUrl: (url: string) => void): Promise<OAuthTokenResponse> {
    const port = await findAvailablePort();
    const redirectUri = `http://localhost:${port}/callback`;
    const controller = new McpOAuthController(serverName, config, redirectUri);

    // 1. Fetch metadata
    const metadataUrl = new URL(config.url);
    metadataUrl.pathname = "/.well-known/mcp-oauth-metadata"; // Mocked discovery
    const metadata: OAuthMetadata = (await axios.get(metadataUrl.toString())).data;

    const authUrl = await controller.getAuthorizationUrl(metadata);
    onOpenUrl(authUrl);

    return new Promise((resolve, reject) => {
        const server = createServer(async (req, res) => {
            const url = new URL(req.url || "", `http://localhost:${port}`);
            if (url.pathname === "/callback") {
                const code = url.searchParams.get("code");
                const state = url.searchParams.get("state");
                const error = url.searchParams.get("error");

                if (error) {
                    res.writeHead(400);
                    res.end(`OAuth Error: ${error}`);
                    server.close();
                    reject(new Error(`OAuth Error: ${error}`));
                    return;
                }

                if (state !== await controller.getState()) {
                    res.writeHead(400);
                    res.end("State mismatch");
                    server.close();
                    reject(new Error("OAuth state mismatch"));
                    return;
                }

                res.writeHead(200, { "Content-Type": "text/html" });
                res.end("<h1>Auth Success</h1><p>You can close this window now.</p>");
                server.close();

                if (code) {
                    try {
                        const tokens = await controller.exchangeCode(code);
                        await controller.saveTokens(tokens);
                        resolve(tokens);
                    } catch (err) {
                        reject(err);
                    }
                }
            }
        });

        server.listen(port);

        // Timeout
        setTimeout(() => {
            server.close();
            reject(new Error("OAuth flow timed out"));
        }, 300000);
    });
}
