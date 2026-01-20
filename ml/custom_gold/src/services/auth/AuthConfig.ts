import { isOauthActive, getStoredOauthTokens } from "./oauthManager.js";
import { getApiKeyWithSource } from "./apiKeyManager.js";

export function getEnvHelper() {
    return {
        BASE_API_URL: process.env.ANTHROPIC_CONSOLE_BASE_URL || "https://console.anthropic.com",
        CLAUDEAI_SUCCESS_URL: "https://claude.ai/login_callback",
        CONSOLE_SUCCESS_URL: "https://console.anthropic.com/settings/keys"
    };
}

export async function getAuthHeaders(): Promise<{ headers: Record<string, string> } | { error: string }> {
    if (isOauthActive()) {
        const tokens = getStoredOauthTokens();
        if (tokens?.accessToken) {
            return {
                headers: {
                    "Authorization": `Bearer ${tokens.accessToken}`
                }
            };
        }
    }

    const { key } = getApiKeyWithSource();
    if (key) {
        return {
            headers: {
                "x-api-key": key
            }
        };
    }

    return { error: "No authentication credentials found" };
}
