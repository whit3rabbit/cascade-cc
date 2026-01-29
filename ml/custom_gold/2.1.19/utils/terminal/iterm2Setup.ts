/**
 * File: src/utils/terminal/iterm2Setup.ts
 * Role: Utilities for configuring and verifying it2 integration in iTerm2.
 */

import { homedir } from "node:os";
import { executeBashCommand, BashResult } from "../shared/bashUtils.js";
import { getSettings, updateSettings } from "../../services/config/SettingsService.js";
import { track } from "../../services/telemetry/Telemetry.js";

/**
 * Checks for available Python package managers.
 */
export async function findPythonPackageManager(): Promise<string | null> {
    if ((await executeBashCommand("which uv")).exitCode === 0) {
        return "uvx";
    }
    if ((await executeBashCommand("which pipx")).exitCode === 0) {
        return "pipx";
    }
    if ((await executeBashCommand("which pip")).exitCode === 0) {
        return "pip";
    }
    if ((await executeBashCommand("which pip3")).exitCode === 0) {
        return "pip";
    }
    return null;
}

/**
 * Checks if the 'it2' command is available.
 */
export async function isIt2Installed(): Promise<boolean> {
    return (await executeBashCommand("which it2")).exitCode === 0;
}

/**
 * Installs the 'it2' CLI tool using the specified package manager.
 */
export async function installIt2(packageManager: string): Promise<{ success: boolean; error?: string }> {
    console.log(`[it2Setup] Installing it2 using ${packageManager}...`);
    let res: BashResult;

    const cwd = homedir();
    switch (packageManager) {
        case "uvx":
            res = await executeBashCommand("uv tool install it2", { cwd });
            break;
        case "pipx":
            res = await executeBashCommand("pipx install it2", { cwd });
            break;
        case "pip":
            res = await executeBashCommand("pip install --user it2", { cwd });
            if (res.exitCode !== 0) {
                res = await executeBashCommand("pip3 install --user it2", { cwd });
            }
            break;
        default:
            return { success: false, error: `Unknown package manager: ${packageManager}` };
    }

    if (res.exitCode !== 0) {
        const error = res.stderr || "Unknown installation error";
        track("tengu_it2_install_failed", { packageManager, error });
        return { success: false, error };
    }

    console.log("[it2Setup] it2 installed successfully");
    return { success: true };
}

/**
 * Verifies if it2 is operational.
 */
export async function verifyIt2Setup(): Promise<{ success: boolean; error?: string; needsPythonApiEnabled?: boolean }> {
    if (!(await isIt2Installed())) {
        return { success: false, error: "it2 CLI not found" };
    }

    const res = await executeBashCommand("it2 session list");
    if (res.exitCode !== 0) {
        const out = res.stderr.toLowerCase();
        if (out.includes("api") || out.includes("python") || out.includes("connection refused")) {
            return { success: false, error: "Python API not enabled", needsPythonApiEnabled: true };
        }
        return { success: false, error: res.stderr || "Communication failed" };
    }

    return { success: true };
}

/**
 * Returns setup instructions for iTerm2 Python API.
 */
export function getIt2SetupInstructions(): string[] {
    return [
        "Enable Python API in iTerm2:",
        "",
        "  iTerm2 → Settings → General → Magic → Enable Python API",
        "",
        "Restart iTerm2 after enabling."
    ];
}

/**
 * Marks it2 setup as complete in settings.
 */
export function markIt2SetupAsComplete(): void {
    updateSettings({ iterm2It2SetupComplete: true });
}

/**
 * Sets preference for tmux vs iTerm2.
 */
export function setPreferTmuxOverIterm2(prefer: boolean): void {
    updateSettings({ preferTmuxOverIterm2: prefer });
}

/**
 * Checks if tmux is preferred.
 */
export function isTmuxPreferredOverIterm2(): boolean {
    return getSettings().preferTmuxOverIterm2 === true;
}
