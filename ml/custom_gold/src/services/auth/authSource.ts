import { toBoolean } from "../../utils/settings/runtimeSettingsAndAuth.js";

/**
 * Detects the source of the authentication token.
 * Deobfuscated from Ei in chunk_174.ts.
 */
export function getAuthTokenSource(): { source: string; hasToken: boolean } {
    if (process.env.ANTHROPIC_AUTH_TOKEN) {
        return {
            source: "ANTHROPIC_AUTH_TOKEN",
            hasToken: true
        };
    }
    if (process.env.CLAUDE_CODE_OAUTH_TOKEN) {
        return {
            source: "CLAUDE_CODE_OAUTH_TOKEN",
            hasToken: true
        };
    }

    // Placeholder for check for file descriptor-based OAuth tokens
    if (process.env.CLAUDE_CODE_OAUTH_TOKEN_FD) {
        return {
            source: "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
            hasToken: true
        };
    }

    // Placeholder for API key helper (shell command etc)
    if (process.env.CLAUDE_CODE_API_KEY_HELPER) {
        return {
            source: "apiKeyHelper",
            hasToken: true
        };
    }

    // Placeholder for Claude.ai session checks
    if (process.env.CLAUDE_AI_SESSION_TOKEN) {
        return {
            source: "claude.ai",
            hasToken: true
        };
    }

    return {
        source: "none",
        hasToken: false
    };
}

/**
 * Checks if user interaction is needed for authentication.
 * Deobfuscated from pU in chunk_174.ts.
 */
export function isInteractiveAuthRequired(): boolean {
    const useBedrock = toBoolean(process.env.CLAUDE_CODE_USE_BEDROCK);
    const useVertex = toBoolean(process.env.CLAUDE_CODE_USE_VERTEX);
    const useFoundry = toBoolean(process.env.CLAUDE_CODE_USE_FOUNDRY);

    if (useBedrock || useVertex || useFoundry) return false;

    const authToken = process.env.ANTHROPIC_AUTH_TOKEN;
    const apiKeyFileDescriptor = process.env.CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR;
    // This would normally check an API key helper too

    const { source, hasToken } = getAuthTokenSource();

    // If we have a direct key via env or FD, or a helper, interaction might not be needed
    if (authToken || apiKeyFileDescriptor || source === "ANTHROPIC_API_KEY" || source === "apiKeyHelper") {
        // Unless we are remote but interaction is explicitly enabled
        if (!toBoolean(process.env.CLAUDE_CODE_REMOTE)) return false;
    }

    return true;
}

/**
 * Checks if any API key or token is configured.
 * Deobfuscated from LsQ in chunk_174.ts.
 */
export function isApiKeyConfigured(): boolean {
    const { hasToken, source } = getAuthTokenSource();
    return hasToken && source !== "none";
}
