import { z } from "zod";
import { generatePkceChallenge } from "../../utils/shared/pkce.js";

// --- Schemas ---

const UrlSchema = z.string().url().refine((url) => {
    try {
        const u = new URL(url);
        return u.protocol !== "javascript:" && u.protocol !== "data:" && u.protocol !== "vbscript:";
    } catch { return false; }
}, { message: "URL cannot use javascript:, data:, or vbscript: scheme" });

export const AuthorizationServerMetadataSchema = z.object({
    issuer: z.string().optional(),
    authorization_endpoint: UrlSchema,
    token_endpoint: UrlSchema,
    registration_endpoint: UrlSchema.optional(),
    scopes_supported: z.array(z.string()).optional(),
    response_types_supported: z.array(z.string()),
    token_endpoint_auth_methods_supported: z.array(z.string()).optional(),
    code_challenge_methods_supported: z.array(z.string()).optional(),
}).passthrough();

export const TokenResponseSchema = z.object({
    access_token: z.string(),
    id_token: z.string().optional(),
    token_type: z.string(),
    expires_in: z.number().optional(), // Zod coerce?
    scope: z.string().optional(),
    refresh_token: z.string().optional()
});

export type AuthorizationServerMetadata = z.infer<typeof AuthorizationServerMetadataSchema>;
export type TokenResponse = z.infer<typeof TokenResponseSchema>;

// --- Discovery ---

export async function discoverAuthorizationServer(issuer: string, fetchFn: typeof fetch = fetch): Promise<AuthorizationServerMetadata | undefined> {
    const issuerUrl = new URL(issuer);
    // Ensure no trailing slash for well-known construction, but standard says base + .well-known
    // logic from dr3: search .well-known/oauth-authorization-server then .well-known/openid-configuration

    // Normalize path
    const paths = [
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration"
    ];

    // If issuer has path, append there too?
    // The chunk logic (dr3) tries multiple locations.

    const candidates = [
        new URL(".well-known/oauth-authorization-server", issuerUrl),
        new URL(".well-known/openid-configuration", issuerUrl)
    ];

    if (issuerUrl.pathname !== "/") {
        candidates.push(new URL(`${issuerUrl.pathname}/.well-known/oauth-authorization-server`.replace('//', '/'), issuerUrl));
        candidates.push(new URL(`${issuerUrl.pathname}/.well-known/openid-configuration`.replace('//', '/'), issuerUrl));
    }

    for (const url of candidates) {
        try {
            const resp = await fetchFn(url.toString(), { headers: { "Accept": "application/json" } });
            if (resp.ok) {
                const json = await resp.json();
                return AuthorizationServerMetadataSchema.parse(json);
            }
        } catch { }
    }
    return undefined;
}


// --- Authorization ---

export interface AuthRequestParams {
    authorizationEndpoint: string;
    clientId: string;
    redirectUri: string;
    scope?: string;
    state?: string;
    resource?: string;
    codeChallengeMethod?: string;
}

export async function createAuthorizationRequest(params: AuthRequestParams) {
    const { authorizationEndpoint, clientId, redirectUri, scope, state, resource } = params;
    const pkce = await generatePkceChallenge();

    const url = new URL(authorizationEndpoint);
    url.searchParams.set("response_type", "code");
    url.searchParams.set("client_id", clientId);
    url.searchParams.set("redirect_uri", redirectUri);
    url.searchParams.set("code_challenge", pkce.code_challenge);
    url.searchParams.set("code_challenge_method", "S256");

    if (scope) url.searchParams.set("scope", scope);
    if (state) url.searchParams.set("state", state);
    if (resource) url.searchParams.set("resource", resource);

    if (scope?.includes("offline_access")) {
        url.searchParams.set("prompt", "consent");
    }

    return {
        url: url.toString(),
        codeVerifier: pkce.code_verifier
    };
}

// --- Token Exchange ---

export interface TokenRequestParams {
    tokenEndpoint: string;
    clientId: string;
    clientSecret?: string;
    code: string;
    codeVerifier: string;
    redirectUri: string;
    fetchFn?: typeof fetch;
}

export async function exchangeCodeForToken(params: TokenRequestParams): Promise<TokenResponse> {
    const { tokenEndpoint, clientId, clientSecret, code, codeVerifier, redirectUri, fetchFn = fetch } = params;

    const body = new URLSearchParams();
    body.set("grant_type", "authorization_code");
    body.set("code", code);
    body.set("code_verifier", codeVerifier);
    body.set("redirect_uri", redirectUri);
    body.set("client_id", clientId);
    if (clientSecret) body.set("client_secret", clientSecret); // Basic auth preferred if strictly following OAuth?
    // The chunk supports multiple auth methods. Simple POST body is common.

    const headers = new Headers({
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    });

    // Simple client_secret_post

    const resp = await fetchFn(tokenEndpoint, {
        method: "POST",
        headers,
        body
    });

    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Token request failed: ${resp.status} ${text}`);
    }

    const json = await resp.json();
    return TokenResponseSchema.parse(json);
}

export async function refreshAccessToken(
    tokenEndpoint: string,
    refreshToken: string,
    clientId: string,
    clientSecret?: string,
    fetchFn: typeof fetch = fetch
): Promise<TokenResponse> {
    const body = new URLSearchParams();
    body.set("grant_type", "refresh_token");
    body.set("refresh_token", refreshToken);
    body.set("client_id", clientId);
    if (clientSecret) body.set("client_secret", clientSecret);

    const headers = new Headers({
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    });

    const resp = await fetchFn(tokenEndpoint, {
        method: "POST",
        headers,
        body
    });

    if (!resp.ok) {
        throw new Error(`Refresh failed: ${resp.status}`);
    }

    return TokenResponseSchema.parse(await resp.json());
}
