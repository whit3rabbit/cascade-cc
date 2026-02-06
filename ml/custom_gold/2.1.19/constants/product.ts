import { getOAuthConfig } from "./oauth.js";

export const PRODUCT_NAME = "Claude Code";

/**
 * ID for the Claude in Chrome browser extension.
 */
export const BROWSER_EXTENSION_ID = "fcoeoabgfenejglbffodgkkbkcdhcgfn";

/**
 * List of allowed beta features (Anthropic headers).
 */
export const ALLOWED_BETA_FEATURES = [
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "max-tokens-32k",
    "token-efficient-tools-2024-12-04",
    "oauth-2025-04-20"
];

/**
 * Maps deployment environments to specific feature sets.
 */
export const DEPLOYMENT_ENVIRONMENTS = {
    FIRST_PARTY: "first-party",
    FOUNDRY: "foundry",
    VERTEX: "vertex",
    BEDROCK: "bedrock",
    NATIVE: "native"
} as const;

/**
 * OAuth and Product Constants.
 */
export const HTTP_PROTOCOL = "http://";
export const LOCALHOST = "127.0.0.1";

const oauthConfig = getOAuthConfig();
export const CLIENT_ID = oauthConfig.CLIENT_ID;
export const TOKEN_URL = oauthConfig.TOKEN_URL;
export const ROLES_URL = oauthConfig.ROLES_URL;
export const PROFILE_URL = `${oauthConfig.BASE_API_URL}/api/oauth/profile`;
export const CLAUDE_CLI_PROFILE_URL = `${oauthConfig.BASE_API_URL}/api/claude_cli_profile`;
export const API_KEY_URL = oauthConfig.API_KEY_URL;
export const CLAUDE_AI_AUTHORIZE_URL = oauthConfig.CLAUDE_AI_AUTHORIZE_URL;
export const CONSOLE_AUTHORIZE_URL = oauthConfig.CONSOLE_AUTHORIZE_URL;
export const CONSOLE_SUCCESS_URL = oauthConfig.CONSOLE_SUCCESS_URL;
export const CLAUDEAI_SUCCESS_URL = oauthConfig.CLAUDEAI_SUCCESS_URL;
export const MANUAL_REDIRECT_URL = oauthConfig.MANUAL_REDIRECT_URL;
export const OAUTH_FILE_SUFFIX = oauthConfig.OAUTH_FILE_SUFFIX;
export const MCP_PROXY_URL = oauthConfig.MCP_PROXY_URL;
export const MCP_PROXY_PATH = oauthConfig.MCP_PROXY_PATH;
export const OAUTH_BETA_HEADER = "oauth-2025-04-20";

export const PRODUCT_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude.";
export const PRODUCT_SYSTEM_PROMPT_SDK = "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.";
export const AGENT_SYSTEM_PROMPT = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";
