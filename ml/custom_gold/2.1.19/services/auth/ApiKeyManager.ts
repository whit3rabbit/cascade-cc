/**
 * File: src/services/auth/ApiKeyManager.ts
 * Role: Manages API key based authentication and OAuth-related error factories.
 */

/**
 * Singleton implementation for managing API key based authentication.
 */
class ApiKeyManagerImpl {
    private apiKey: string | null = null;

    /**
     * Gets the current API key.
     * @returns {string | null}
     */
    getApiKey(): string | null {
        return this.apiKey;
    }

    /**
     * Sets the API key.
     * @param {string | null} key
     */
    setApiKey(key: string | null): void {
        this.apiKey = key;
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
