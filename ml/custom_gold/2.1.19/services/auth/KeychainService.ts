/**
 * File: src/services/auth/KeychainService.ts
 * Role: Manages persistent credentials across macOS, Linux, and Windows.
 */

import { spawnSync } from 'node:child_process';
import { userInfo } from 'node:os';
import { EnvService } from '../config/EnvService.js';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { existsSync, readFileSync, writeFileSync, mkdirSync, unlinkSync, chmodSync } from 'node:fs';

const CONFIG_DIR = process.env.CLAUDE_CONFIG_DIR || join(homedir(), '.claude');
const PLAINTEXT_STORAGE_PATH = join(CONFIG_DIR, '.credentials.json');

/**
 * In-memory cache to avoid repeated slow system calls.
 */
const tokenCache: Record<string, string | null> = {};

/**
 * Generic command runner.
 */
function runCommand(command: string, args: string[], input: string | null = null): { code: number | null, stdout: string, stderr: string } {
    try {
        const result = spawnSync(command, args, {
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
    return EnvService.get("USER") || userInfo().username || "claude-code-user";
}

export interface KeychainInterface {
    isAvailable(): boolean;
    readToken(serviceName: string): string | null;
    saveToken(serviceName: string, token: string): boolean;
    deleteToken(serviceName: string): boolean;
}

/**
 * Plaintext fallback storage (chunk713: createPlaintextStorage logic).
 */
const PlaintextStorage = {
    read(serviceName: string): string | null {
        if (existsSync(PLAINTEXT_STORAGE_PATH)) {
            try {
                const data = JSON.parse(readFileSync(PLAINTEXT_STORAGE_PATH, 'utf8'));
                return data[serviceName] || null;
            } catch {
                return null;
            }
        }
        return null;
    },
    save(serviceName: string, token: string): boolean {
        try {
            if (!existsSync(CONFIG_DIR)) {
                mkdirSync(CONFIG_DIR, { recursive: true });
            }
            let data: Record<string, string> = {};
            if (existsSync(PLAINTEXT_STORAGE_PATH)) {
                try {
                    data = JSON.parse(readFileSync(PLAINTEXT_STORAGE_PATH, 'utf8'));
                } catch {
                    data = {};
                }
            }
            data[serviceName] = token;
            writeFileSync(PLAINTEXT_STORAGE_PATH, JSON.stringify(data, null, 2), { encoding: 'utf8' });
            chmodSync(PLAINTEXT_STORAGE_PATH, 0o600);
            return true;
        } catch {
            return false;
        }
    },
    delete(serviceName: string): boolean {
        if (existsSync(PLAINTEXT_STORAGE_PATH)) {
            try {
                const data = JSON.parse(readFileSync(PLAINTEXT_STORAGE_PATH, 'utf8'));
                if (data[serviceName]) {
                    delete data[serviceName];
                    writeFileSync(PLAINTEXT_STORAGE_PATH, JSON.stringify(data, null, 2), { encoding: 'utf8' });
                }
                return true;
            } catch {
                return false;
            }
        }
        return true;
    }
};

/**
 * Service for cross-platform Keychain operations.
 */
export const KeychainService: KeychainInterface = {
    isAvailable(): boolean {
        const platform = process.platform;
        if (platform === "darwin") {
            // Gold standard check: 0 or 36 or 128 implies it's "available" but might be locked
            const res = runCommand("security", ["show-keychain-info"]);
            return res.code === 0 || res.code === 36 || res.code === 128;
        }
        if (platform === "win32") {
            const res = runCommand("powershell", ["-Command", "[Windows.Security.Credentials.PasswordVault, Windows.Security.Credentials, ContentType=WindowsRuntime] > $null"]);
            return res.code === 0;
        }
        if (platform === "linux") {
            return runCommand("which", ["secret-tool"]).code === 0;
        }
        return true; // Plaintext fallback is always available
    },

    readToken(serviceName: string): string | null {
        if (tokenCache[serviceName] !== undefined) {
            return tokenCache[serviceName];
        }

        const user = getUserName();
        const platform = process.platform;
        let token: string | null = null;
        let useFallback = false;

        if (platform === "darwin") {
            let res = runCommand("security", ["find-generic-password", "-a", user, "-w", "-s", serviceName]);
            if (res.code !== 0 && res.code !== 128) {
                // Fallback: try without specifying account
                const fallbackRes = runCommand("security", ["find-generic-password", "-w", "-s", serviceName]);
                if (fallbackRes.code === 0) {
                    res = fallbackRes;
                }
            }

            if (res.code === 0) {
                token = res.stdout;
            } else {
                // Return codes: 36 = access denied/locked, 128 = user canceled
                if (res.code === 36 || res.stderr.includes("errSecAuthFailed")) {
                    console.warn(`Keychain access denied for "${serviceName}" on macOS. Falling back to plaintext storage.`);
                    useFallback = true;
                } else if (res.code === 128) {
                    console.warn(`Keychain access canceled by user for "${serviceName}". Falling back to plaintext storage.`);
                    useFallback = true;
                }
            }
        }
        else if (platform === "win32") {
            const script = `$vault = New-Object Windows.Security.Credentials.PasswordVault; try { $c = $vault.Retrieve("${serviceName}", "${user}"); $c.FillPassword(); $c.Password } catch { exit 1 }`;
            const res = runCommand("powershell", ["-Command", script]);
            if (res.code === 0) token = res.stdout;
        } else if (platform === "linux") {
            const res = runCommand("secret-tool", ["lookup", "service", serviceName, "account", user]);
            if (res.code === 0) {
                token = res.stdout;
            } else {
                useFallback = true;
            }
        } else {
            useFallback = true;
        }

        if (useFallback || (!token && platform !== "win32")) {
            token = PlaintextStorage.read(serviceName);
        }

        tokenCache[serviceName] = token;
        return token;
    },

    saveToken(serviceName: string, token: string): boolean {
        const user = getUserName();
        const platform = process.platform;
        let success = false;
        let useFallback = false;

        if (platform === "darwin") {
            const tokenHex = Buffer.from(token, "utf-8").toString("hex");
            const res = runCommand("security", ["add-generic-password", "-U", "-a", user, "-s", serviceName, "-X", tokenHex]);
            if (res.code === 0) {
                success = true;
            } else if (res.code === 36 || res.code === 128) {
                useFallback = true;
            }
        } else if (platform === "win32") {
            const script = `$vault = New-Object Windows.Security.Credentials.PasswordVault; $c = New-Object Windows.Security.Credentials.PasswordCredential("${serviceName}", "${user}", "${token}"); $vault.Add($c)`;
            const res = runCommand("powershell", ["-Command", script]);
            success = res.code === 0;
        } else if (platform === "linux") {
            const res = runCommand("secret-tool", ["store", "--label=Claude Code Token", "service", serviceName, "account", user], token);
            if (res.code === 0) {
                success = true;
            } else {
                useFallback = true;
            }
        } else {
            useFallback = true;
        }

        if (useFallback) {
            success = PlaintextStorage.save(serviceName, token);
        }

        if (success) {
            tokenCache[serviceName] = token;
        }
        return success;
    },

    deleteToken(serviceName: string): boolean {
        delete tokenCache[serviceName];
        const user = getUserName();
        const platform = process.platform;
        let success = false;

        if (platform === "darwin") {
            const res = runCommand("security", ["delete-generic-password", "-a", user, "-s", serviceName]);
            success = res.code === 0;
        } else if (platform === "win32") {
            const script = `$vault = New-Object Windows.Security.Credentials.PasswordVault; try { $c = $vault.Retrieve("${serviceName}", "${user}"); $vault.Remove($c) } catch { exit 1 }`;
            const res = runCommand("powershell", ["-Command", script]);
            success = res.code === 0;
        } else if (platform === "linux") {
            const res = runCommand("secret-tool", ["clear", "service", serviceName, "account", user]);
            success = res.code === 0;
        }

        // Always attempt to delete from plaintext fallback
        const fallbackSuccess = PlaintextStorage.delete(serviceName);
        return success || fallbackSuccess;
    }
};
