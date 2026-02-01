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
export const CLIENT_ID = "claude-code-cli";
export const TOKEN_URL = "https://platform.claude.com/v1/oauth/token";
export const ROLES_URL = "https://api.anthropic.com/api/oauth/claude_cli/roles";
export const PROFILE_URL = "https://api.anthropic.com/api/oauth/profile";
export const CLAUDE_CLI_PROFILE_URL = "https://api.anthropic.com/api/claude_cli_profile";
export const API_KEY_URL = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key";
export const CLAUDE_AI_AUTHORIZE_URL = "https://claude.ai/oauth/authorize";
export const CONSOLE_AUTHORIZE_URL = "https://platform.claude.com/oauth/authorize";
export const MANUAL_REDIRECT_URL = "https://platform.claude.com/oauth/code/callback";
export const OAUTH_BETA_HEADER = "oauth-2025-04-20";

export const PRODUCT_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude.";
export const PRODUCT_SYSTEM_PROMPT_SDK = "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.";
export const AGENT_SYSTEM_PROMPT = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";

