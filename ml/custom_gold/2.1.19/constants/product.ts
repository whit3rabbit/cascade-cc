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
    "token-efficient-tools-2024-12-04"
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
export const CLIENT_ID = "claude-code-cli";
export const TOKEN_URL = "https://api.anthropic.com/api/auth/token";
export const ROLES_URL = "https://api.anthropic.com/api/auth/roles";
export const API_KEY_URL = "https://api.anthropic.com/api/auth/api_key";
export const CLAUDE_AI_AUTHORIZE_URL = "https://claude.ai/login_oauth";
export const CONSOLE_AUTHORIZE_URL = "https://console.anthropic.com/login_oauth";
export const MANUAL_REDIRECT_URL = "urn:ietf:wg:oauth:2.0:oob";
