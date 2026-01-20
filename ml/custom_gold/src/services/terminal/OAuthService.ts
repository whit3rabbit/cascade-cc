
// Logic from chunk_96.ts (OAuth & Profile Management)

import { getConfig, updateConfig } from "./ConfigService.js";

/**
 * Handles Anthropic OAuth flow, token management, and profile fetching.
 */
export const OAuthService = {
    getAuthorizeUrl(options: {
        state: string,
        codeChallenge: string,
        port: number,
        isManual?: boolean
    }) {
        const url = new URL("https://auth.anthropic.com/authorize");
        url.searchParams.append("client_id", "claude-code");
        url.searchParams.append("response_type", "code");
        url.searchParams.append("code_challenge", options.codeChallenge);
        url.searchParams.append("code_challenge_method", "S256");
        url.searchParams.append("state", options.state);
        url.searchParams.append("redirect_uri", options.isManual ? "https://claude.ai/code/callback" : `http://localhost:${options.port}/callback`);
        return url.toString();
    },

    async exchangeCodeForToken(code: string, verifier: string) {
        console.log("Exchanging authorization code for tokens...");
        // POST to https://api.anthropic.com/oauth/token
        return { accessToken: "sk-ant-...", refreshToken: "...", expiresAt: Date.now() + 3600000 };
    },

    async refreshToken(token: string) {
        console.log("Refreshing access token...");
        // Logic to refresh and update config
        return { accessToken: "new-sk-...", refreshToken: "...", expiresAt: Date.now() + 3600000 };
    },

    async fetchUserProfile(accessToken: string) {
        // GET to https://api.anthropic.com/api/oauth/profile
        return {
            displayName: "User",
            email: "user@example.com",
            orgName: "Anthropic",
            tier: "pro" as const
        };
    }
};

/**
 * Utility to detect the current inference provider based on env vars.
 */
export function getInferenceProvider(): "firstParty" | "bedrock" | "vertex" | "foundry" {
    if (process.env.CLAUDE_CODE_USE_BEDROCK) return "bedrock";
    if (process.env.CLAUDE_CODE_USE_VERTEX) return "vertex";
    if (process.env.CLAUDE_CODE_USE_FOUNDRY) return "foundry";
    return "firstParty";
}
