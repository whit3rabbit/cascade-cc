/**
 * File: src/constants/oauth.ts
 * Role: OAuth configuration and scope definitions aligned with 2.1.19.
 */

export type OAuthEnvironment = "local" | "staging" | "prod";

export interface OAuthConfig {
    BASE_API_URL: string;
    CONSOLE_AUTHORIZE_URL: string;
    CLAUDE_AI_AUTHORIZE_URL: string;
    TOKEN_URL: string;
    API_KEY_URL: string;
    ROLES_URL: string;
    CONSOLE_SUCCESS_URL: string;
    CLAUDEAI_SUCCESS_URL: string;
    MANUAL_REDIRECT_URL: string;
    CLIENT_ID: string;
    OAUTH_FILE_SUFFIX: string;
    MCP_PROXY_URL: string;
    MCP_PROXY_PATH: string;
}

export const OAUTH_SCOPE_INFERENCE = "user:inference";
export const OAUTH_SCOPE_CREATE_API_KEY = "org:create_api_key";
export const OAUTH_SCOPE_PROFILE = "user:profile";
export const OAUTH_SCOPE_SESSIONS = "user:sessions:claude_code";
export const OAUTH_SCOPE_MCP_SERVERS = "user:mcp_servers";

export const OAUTH_SCOPES_API_KEY = [OAUTH_SCOPE_CREATE_API_KEY, OAUTH_SCOPE_PROFILE];
export const OAUTH_SCOPES_DEFAULT = [OAUTH_SCOPE_PROFILE, OAUTH_SCOPE_INFERENCE, OAUTH_SCOPE_SESSIONS, OAUTH_SCOPE_MCP_SERVERS];
export const OAUTH_SCOPES_ALL = Array.from(new Set([...OAUTH_SCOPES_API_KEY, ...OAUTH_SCOPES_DEFAULT]));

export function parseOAuthScopes(scope: string | null | undefined): string[] {
    return scope?.split(" ").filter(Boolean) ?? [];
}

export function hasInferenceScope(scopes: string[] | string | null | undefined): boolean {
    if (!scopes) return false;
    const list = Array.isArray(scopes) ? scopes : parseOAuthScopes(scopes);
    return list.includes(OAUTH_SCOPE_INFERENCE);
}

function getOAuthEnvironment(): OAuthEnvironment {
    const raw = process.env.CLAUDE_CODE_OAUTH_ENV;
    if (raw === "local" || raw === "staging" || raw === "prod") {
        return raw;
    }
    return "prod";
}

const PROD_OAUTH_CONFIG: OAuthConfig = {
    BASE_API_URL: "https://api.anthropic.com",
    CONSOLE_AUTHORIZE_URL: "https://platform.claude.com/oauth/authorize",
    CLAUDE_AI_AUTHORIZE_URL: "https://claude.ai/oauth/authorize",
    TOKEN_URL: "https://platform.claude.com/v1/oauth/token",
    API_KEY_URL: "https://api.anthropic.com/api/oauth/claude_cli/create_api_key",
    ROLES_URL: "https://api.anthropic.com/api/oauth/claude_cli/roles",
    CONSOLE_SUCCESS_URL: "https://platform.claude.com/buy_credits?returnUrl=/oauth/code/success%3Fapp%3Dclaude-code",
    CLAUDEAI_SUCCESS_URL: "https://platform.claude.com/oauth/code/success?app=claude-code",
    MANUAL_REDIRECT_URL: "https://platform.claude.com/oauth/code/callback",
    CLIENT_ID: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
    OAUTH_FILE_SUFFIX: "",
    MCP_PROXY_URL: "https://mcp-proxy.anthropic.com",
    MCP_PROXY_PATH: "/v1/mcp/{server_id}"
};

const LOCAL_OAUTH_CONFIG: OAuthConfig = {
    BASE_API_URL: "http://localhost:3000",
    CONSOLE_AUTHORIZE_URL: "http://localhost:3000/oauth/authorize",
    CLAUDE_AI_AUTHORIZE_URL: "http://localhost:4000/oauth/authorize",
    TOKEN_URL: "http://localhost:3000/v1/oauth/token",
    API_KEY_URL: "http://localhost:3000/api/oauth/claude_cli/create_api_key",
    ROLES_URL: "http://localhost:3000/api/oauth/claude_cli/roles",
    CONSOLE_SUCCESS_URL: "http://localhost:3000/buy_credits?returnUrl=/oauth/code/success%3Fapp%3Dclaude-code",
    CLAUDEAI_SUCCESS_URL: "http://localhost:3000/oauth/code/success?app=claude-code",
    MANUAL_REDIRECT_URL: "https://console.staging.ant.dev/oauth/code/callback",
    CLIENT_ID: "22422756-60c9-4084-8eb7-27705fd5cf9a",
    OAUTH_FILE_SUFFIX: "-local-oauth",
    MCP_PROXY_URL: "http://localhost:8205",
    MCP_PROXY_PATH: "/v1/toolbox/shttp/mcp/{server_id}"
};

let cachedConfig: OAuthConfig | null = null;

export function getOAuthConfig(): OAuthConfig {
    if (cachedConfig) return cachedConfig;

    const env = getOAuthEnvironment();
    const baseConfig = env === "local"
        ? LOCAL_OAUTH_CONFIG
        : env === "staging"
            ? PROD_OAUTH_CONFIG
            : PROD_OAUTH_CONFIG;

    const clientIdOverride = process.env.CLAUDE_CODE_OAUTH_CLIENT_ID;
    cachedConfig = clientIdOverride ? { ...baseConfig, CLIENT_ID: clientIdOverride } : baseConfig;
    return cachedConfig;
}
