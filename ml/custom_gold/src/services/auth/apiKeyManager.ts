import { execSync } from "node:child_process";
import { memoize } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { toBoolean } from "../../utils/settings/runtimeSettingsAndAuth.js";
import { deleteMacosKeychainPassword } from "./awsCredentials.js";

const DEFAULT_HELPER_TTL_MS = 300000; // 5 minutes

/**
 * Retrieves the API key and identifies its source.
 */
export function getApiKeyWithSource(options: { skipRetrievingKeyFromApiKeyHelper?: boolean } = {}): { key: string | null; source: string } {
    // 1. Check environment variable
    if (process.env.ANTHROPIC_API_KEY) {
        return {
            key: process.env.ANTHROPIC_API_KEY,
            source: "ANTHROPIC_API_KEY"
        };
    }

    // 2. Check for configured helper
    const helper = process.env.CLAUDE_CODE_API_KEY_HELPER;
    if (helper) {
        if (options.skipRetrievingKeyFromApiKeyHelper) {
            return { key: null, source: "apiKeyHelper" };
        }
        const key = (getApiKeyFromHelper as any)(helper);
        if (key) return { key, source: "apiKeyHelper" };
    }

    // 3. Check for macOS Keychain / Managed key
    const managedKey = getManagedApiKey();
    if (managedKey) return managedKey;

    return {
        key: null,
        source: "none"
    };
}

/**
 * Runs a shell command helper to get the API key.
 */
export const getApiKeyFromHelper = memoize((helperCommand: string): string | null => {
    try {
        const output = execSync(helperCommand, { encoding: "utf8", timeout: 10000 }).trim();
        return output || null;
    } catch (err) {
        console.error(`Error running apiKeyHelper: ${err instanceof Error ? err.message : String(err)}`);
        return null;
    }
}, (helperCommand: string) => {
    // Memoize based on command and time (TTL)
    const ttl = parseInt(process.env.CLAUDE_CODE_API_KEY_HELPER_TTL_MS || String(DEFAULT_HELPER_TTL_MS));
    const timeKey = Math.floor(Date.now() / ttl);
    return `${helperCommand}:${timeKey}`;
});

/**
 * Checks if the API key helper is trusted.
 */
export function isApiKeyHelperTrusted(): boolean {
    const helper = process.env.CLAUDE_CODE_API_KEY_HELPER;
    if (!helper) return false;
    // This would check against project or local settings trust lists
    return true;
}

/**
 * Saves the API key to localized storage (macOS Keychain or config).
 */
export function saveApiKey(key: string): void {
    // Basic validation
    if (!/^[a-zA-Z0-9-_]+$/.test(key)) {
        throw new Error("Invalid API key format.");
    }

    // Delete old one if exists
    deleteApiKey();

    if (process.platform === "darwin") {
        try {
            // simplified keychain save logic
            const service = "claude-code";
            const account = process.env.USER || "default";
            execSync(`security add-generic-password -U -a "${account}" -s "${service}" -w "${key}"`, { stdio: "ignore" });
            return;
        } catch (err) {
            console.error("Failed to save API key to Keychain.");
        }
    }

    // Fallback: save to local config (not implemented here)
}

/**
 * Deletes the API key from storage.
 */
export function deleteApiKey(): void {
    if (process.platform === "darwin") {
        const service = "claude-code";
        const account = process.env.USER || "default";
        deleteMacosKeychainPassword(account, service);
    }
}

/**
 * Retrieves the managed API key (e.g. from Keychain).
 */
function getManagedApiKey(): { key: string; source: string } | null {
    if (process.platform === "darwin") {
        try {
            const service = "claude-code";
            const account = process.env.USER || "default";
            const key = execSync(`security find-generic-password -a "${account}" -w -s "${service}"`, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
            if (key) return { key, source: "/login managed key" };
        } catch (err) {
            // Not found
        }
    }
    return null;
}
