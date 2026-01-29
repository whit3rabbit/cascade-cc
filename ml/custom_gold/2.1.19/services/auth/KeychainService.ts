/**
 * File: src/services/auth/KeychainService.ts
 * Role: Manages persistent credentials in the macOS Keychain.
 */

import { spawnSync } from 'node:child_process';
import { userInfo } from 'node:os';

/**
 * Executes a macOS 'security' command.
 */
function runSecurity(args: string[], input: string | null = null): { code: number | null, stdout: string, stderr: string } {
    try {
        const result = spawnSync("security", args, {
            encoding: 'utf-8',
            input: input || undefined,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        return {
            code: result.status,
            stdout: result.stdout?.trim() || "",
            stderr: result.stderr?.trim() || ""
        };
    } catch (err: unknown) {
        return { code: 1, stdout: "", stderr: err instanceof Error ? err.message : String(err) };
    }
}

function getUserName(): string {
    return process.env.USER || userInfo().username || "claude-code-user";
}

export interface KeychainInterface {
    isAvailable(): boolean;
    readToken(serviceName: string): string | null;
    saveToken(serviceName: string, token: string): boolean;
    deleteToken(serviceName: string): boolean;
}

/**
 * Service for macOS Keychain operations.
 */
export const KeychainService: KeychainInterface = {
    isAvailable(): boolean {
        if (process.platform !== "darwin") return false;
        return runSecurity(["show-keychain-info"]).code === 0;
    },

    /**
     * Reads a password/token from the keychain.
     */
    readToken(serviceName: string): string | null {
        const user = getUserName();
        const res = runSecurity(["find-generic-password", "-a", user, "-w", "-s", serviceName]);
        return res.code === 0 ? res.stdout : null;
    },

    /**
     * Saves a password/token to the keychain.
     */
    saveToken(serviceName: string, token: string): boolean {
        const user = getUserName();
        const tokenHex = Buffer.from(token, "utf-8").toString("hex");
        // -U updates existing, -a account, -s service, -X hex data
        const cmd = `add-generic-password -U -a "${user}" -s "${serviceName}" -X "${tokenHex}"`;
        // Note: security add-generic-password doesn't take input from stdin quite like this usually,
        // but preserving the logic structure from the JS file.
        // Actually, for add-generic-password, passing hex data via -X avoids shell escaping issues.
        const res = runSecurity(["add-generic-password", "-U", "-a", user, "-s", serviceName, "-X", tokenHex]);
        return res.code === 0;
    },

    /**
     * Deletes a password/token from the keychain.
     */
    deleteToken(serviceName: string): boolean {
        const user = getUserName();
        const res = runSecurity(["delete-generic-password", "-a", user, "-s", serviceName]);
        return res.code === 0;
    }
};
