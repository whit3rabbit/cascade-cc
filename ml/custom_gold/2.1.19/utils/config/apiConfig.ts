/**
 * File: src/utils/config/apiConfig.ts
 * Role: Centralized API configuration and model constants.
 */

export interface ApiConfig {
    BASE_API_URL: string;
}

export const apiConfig: ApiConfig = {
    BASE_API_URL: process.env.CLAUDE_BASE_URL || "https://api.anthropic.com",
};

/**
 * Returns the current API configuration.
 */
export function getApiConfig(): ApiConfig {
    return apiConfig;
}

// Aliases for compatibility
export const b7 = getApiConfig;
