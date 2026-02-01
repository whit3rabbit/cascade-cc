/**
 * File: src/services/config/ConfigStatusService.ts
 * Role: Collects diagnostic information about the current CLI configuration for debugging purposes.
 */

import { ApiKeyManager } from '../auth/ApiKeyManager.js';
import { OAuthService } from '../auth/OAuthService.js';
import { isDemo } from '../../utils/shared/runtime.js';
import { EnvService } from './EnvService.js';

export interface ConfigDetail {
    label: string;
    value: string;
}

/**
 * Returns a list of diagnostic details about the authentication and provider state.
 * 
 * @returns {Promise<ConfigDetail[]>} Array of diagnostic key-value pairs.
 */
export async function getDetailedConfigStatus(): Promise<ConfigDetail[]> {
    const details: ConfigDetail[] = [];

    // 1. Auth Status
    const oauthToken = await OAuthService.getValidToken();
    if (oauthToken) {
        details.push({ label: "Login method", value: "OAuth (Browser)" });
    } else {
        const apiKey = ApiKeyManager.getApiKey();
        if (apiKey) {
            details.push({ label: "Login method", value: "API Key (Manual)" });
        }
    }

    // 2. Demo Status
    if (isDemo()) {
        details.push({ label: "Mode", value: "Demo / Evaluation" });
    }

    // 3. Proxy & Network settings (Stubbed placeholders for now)
    const proxy = EnvService.get("HTTPS_PROXY") || EnvService.get("http_proxy");
    if (proxy) {
        details.push({ label: "Proxy", value: proxy });
    }

    return details;
}
