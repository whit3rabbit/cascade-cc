/**
 * File: src/services/auth/ApiKeyManager.ts
 * Role: Manages API key based authentication and OAuth-related error factories.
 */

import { exec } from "child_process";
import { promisify } from "util";
import { getSettings } from '../config/SettingsService.js';

const execAsync = promisify(exec);

/**
 * Singleton implementation for managing API key based authentication.
 */
class ApiKeyManagerImpl {
    private apiKey: string | null = null;
    private helperKey: string | null = null;
    private lastHelperFetch: number = 0;
    private readonly HELPER_TTL = 5 * 60 * 1000; // 5 minutes

    /**
     * Gets the current API key.
     * @returns {Promise<string | null>}
     */
    async getApiKey(): Promise<string | null> {
        if (this.apiKey) return this.apiKey;

        const envKey = process.env.ANTHROPIC_API_KEY;
        if (envKey) return envKey;

        // Check for helper command in settings
        const settings = getSettings();
        const helperCommand = settings.apiKeyHelper;

        if (helperCommand) {
            const now = Date.now();
            if (this.helperKey && now - this.lastHelperFetch < this.HELPER_TTL) {
                return this.helperKey;
            }

            return await this.getApiKeyFromHelper(helperCommand);
        }

        return null;
    }

    /**
     * Sets the API key.
     * @param {string | null} key
     */
    setApiKey(key: string | null): void {
        this.apiKey = key;
    }

    /**
     * Executes the apiKeyHelper command to retrieve the API key.
     * @param {string} helperCommand The shell command to execute.
     * @param {number} timeoutMs Timeout in milliseconds (default 5 minutes).
     * @returns {Promise<string | null>} The retrieved API key or null on failure.
     */
    async getApiKeyFromHelper(helperCommand: string, timeoutMs: number = 300000): Promise<string | null> {
        try {
            const { stdout } = await execAsync(helperCommand, { timeout: timeoutMs, encoding: 'utf8' });
            const key = stdout.trim();
            if (!key) {
                console.error("apiKeyHelper returned empty output");
                return null;
            }
            this.helperKey = key;
            this.lastHelperFetch = Date.now();
            return key;
        } catch (error) {
            console.error(`Error executing apiKeyHelper: ${error instanceof Error ? error.message : String(error)}`);
            return null;
        }
    }
}

/**
 * Singleton instance of the ApiKeyManager.
 */
export const ApiKeyManager = new ApiKeyManagerImpl();

// Error Factories used by OAuthService

/**
 * Error thrown when a loopback server already exists.
 */
export function createLoopbackServerAlreadyExistsError(): Error {
    return new Error("Loopback server already exists.");
}

/**
 * Error thrown when the redirect URL cannot be loaded.
 */
export function createUnableToLoadRedirectUrlError(): Error {
    return new Error("Unable to load redirect URL.");
}

/**
 * Error thrown when no loopback server exists.
 */
export function createNoLoopbackServerExistsError(): Error {
    return new Error("No loopback server exists.");
}

/**
 * Error thrown when the loopback address type is invalid.
 */
export function createInvalidLoopbackAddressTypeError(): Error {
    return new Error("Invalid loopback address type.");
}
