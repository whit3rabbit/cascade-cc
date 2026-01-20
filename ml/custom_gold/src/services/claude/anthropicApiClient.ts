import { getApiKeyWithSource } from "../auth/apiKeyManager.js";
import { isOauthActive, getStoredOauthTokens } from "../auth/oauthManager.js";

const VERSION = "2.0.76";

// Beta headers from chunk_96/chunk_167
export const BETA_HEADERS = {
    claudeCode: "claude-code-20250219",
    interleavedThinking: "interleaved-thinking-2025-05-14",
    context1m: "context-1m-2025-08-07",
    contextManagement: "context-management-2025-06-27",
    structuredOutputs: "structured-outputs-2025-09-17",
    webSearch: "web-search-2025-03-05",
    toolExamples: "tool-examples-2025-10-29",
    advancedToolUse: "advanced-tool-use-2025-11-20",
    toolSearchTool: "tool-search-tool-2025-10-19"
};

const ALL_BETAS = Object.values(BETA_HEADERS).join(",");

/**
 * Returns the user agent for the Claude CLI.
 */
export function getCliUserAgent(): string {
    return `claude-code/${VERSION}`;
}

/**
 * Returns a more detailed internal user agent including environment info.
 */
export function getInternalUserAgent(): string {
    const sdkVersion = process.env.CLAUDE_AGENT_SDK_VERSION ? `, agent-sdk/${process.env.CLAUDE_AGENT_SDK_VERSION}` : "";
    const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT || "unknown";
    return `claude-cli/${VERSION} (external, ${entrypoint}${sdkVersion})`;
}

/**
 * Generates authentication and beta headers for API calls.
 */
export function getAuthHeaders(): { headers: Record<string, string>, error?: string } {
    if (isOauthActive()) {
        const tokens = (getStoredOauthTokens as any)();
        if (!tokens?.accessToken) return { headers: {}, error: "No OAuth token available" };
        return {
            headers: {
                Authorization: `Bearer ${tokens.accessToken}`,
                "anthropic-beta": ALL_BETAS
            }
        };
    }

    const { key } = getApiKeyWithSource();
    if (!key) return { headers: {}, error: "No API key available" };
    return {
        headers: {
            "x-api-key": key,
            "anthropic-beta": ALL_BETAS
        }
    };
}
