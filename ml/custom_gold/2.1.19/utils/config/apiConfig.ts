/**
 * File: src/utils/config/apiConfig.ts
 * Role: Centralized API configuration and model constants.
 */

export interface ApiConfig {
    BASE_API_URL: string;
}

import { EnvService } from "../../services/config/EnvService.js";

export const apiConfig: ApiConfig = {
    BASE_API_URL: EnvService.get("CLAUDE_BASE_URL"),
};

/**
 * Returns the current API configuration.
 */
export function getApiConfig(): ApiConfig {
    return apiConfig;
}

// Aliases for compatibility
export const b7 = getApiConfig;
