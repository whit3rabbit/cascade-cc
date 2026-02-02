/**
 * File: src/services/auth/KeychainService.ts
 * Role: Manages persistent credentials across macOS, Linux, and Windows.
 */

import { spawnSync } from 'node:child_process';
import { userInfo } from 'node:os';
import { EnvService } from '../config/EnvService.js';

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
 * Service for cross-platform Keychain operations.
 */
export const KeychainService: KeychainInterface = {
    isAvailable(): boolean {
        const platform = process.platform;
        if (platform === "darwin") {
            // Gold standard check: 0 or sometimes 36 implies success/initialized state
            const res = runCommand("security", ["show-keychain-info"]);
            return res.code === 0 || res.code === 36;
        }
        if (platform === "win32") {
            const res = runCommand("powershell", ["-Command", "[Windows.Security.Credentials.PasswordVault, Windows.Security.Credentials, ContentType=WindowsRuntime] > $null"]);
            return res.code === 0;
        }
        if (platform === "linux") {
            return runCommand("which", ["secret-tool"]).code === 0;
        }
        return false;
    },

    readToken(serviceName: string): string | null {
        if (tokenCache[serviceName] !== undefined) {
            return tokenCache[serviceName];
        }

        const user = getUserName();
        const platform = process.platform;
        let token: string | null = null;

        if (platform === "darwin") {
            const res = runCommand("security", ["find-generic-password", "-a", user, "-w", "-s", serviceName]);
            if (res.code === 0) {
                token = res.stdout;
            } else {
                // Return codes: 44 = item not found (silent), 36 = access denied/locked, 128 = user canceled
                if (res.code === 36 || res.stderr.includes("errSecAuthFailed")) {
                    console.warn(`Keychain access denied for "${serviceName}" on macOS.`);
                    console.warn("Guidance: Ensure the terminal has 'Developer Tools' permissions in System Settings > Privacy & Security, or that the binary is properly signed.");
                } else if (res.code === 128) {
                    console.warn(`Keychain access canceled by user for "${serviceName}".`);
                } else if (res.code !== 44) {
                    console.debug(`Keychain error (${res.code}): ${res.stderr}`);
                }
            }
        } else if (platform === "win32") {
            const script = `$vault = New-Object Windows.Security.Credentials.PasswordVault; try { $c = $vault.Retrieve("${serviceName}", "${user}"); $c.FillPassword(); $c.Password } catch { exit 1 }`;
            const res = runCommand("powershell", ["-Command", script]);
            if (res.code === 0) token = res.stdout;
        } else if (platform === "linux") {
            const res = runCommand("secret-tool", ["lookup", "service", serviceName, "account", user]);
            if (res.code === 0) token = res.stdout;
        }

        tokenCache[serviceName] = token;
        return token;
    },

    saveToken(serviceName: string, token: string): boolean {
        const user = getUserName();
        const platform = process.platform;
        let success = false;

        if (platform === "darwin") {
            const tokenHex = Buffer.from(token, "utf-8").toString("hex");
            // Use -i for interactive-like input of the password via -X to avoid command line leaking
            const res = runCommand("security", ["add-generic-password", "-U", "-a", user, "-s", serviceName, "-X", tokenHex]);
            success = res.code === 0;
        } else if (platform === "win32") {
            const script = `$vault = New-Object Windows.Security.Credentials.PasswordVault; $c = New-Object Windows.Security.Credentials.PasswordCredential("${serviceName}", "${user}", "${token}"); $vault.Add($c)`;
            const res = runCommand("powershell", ["-Command", script]);
            success = res.code === 0;
        } else if (platform === "linux") {
            const res = runCommand("secret-tool", ["store", "--label=Claude Code Token", "service", serviceName, "account", user], token);
            success = res.code === 0;
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

        if (platform === "darwin") {
            const res = runCommand("security", ["delete-generic-password", "-a", user, "-s", serviceName]);
            return res.code === 0;
        }
        if (platform === "win32") {
            const script = `$vault = New-Object Windows.Security.Credentials.PasswordVault; try { $c = $vault.Retrieve("${serviceName}", "${user}"); $vault.Remove($c) } catch { exit 1 }`;
            const res = runCommand("powershell", ["-Command", script]);
            return res.code === 0;
        }
        if (platform === "linux") {
            const res = runCommand("secret-tool", ["clear", "service", serviceName, "account", user]);
            return res.code === 0;
        }

        return false;
    }
};
