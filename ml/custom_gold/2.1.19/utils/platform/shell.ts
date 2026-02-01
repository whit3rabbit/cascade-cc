/**
 * File: src/utils/platform/shell.ts
 * Role: Shell detection, normalization, and configuration snapshotting.
 */

import * as os from "node:os";
import * as fs from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { runCommand } from "../process/commandRunner.js";

/**
 * Checks if a file exists and is executable.
 */
export function isExecutable(filePath: string): boolean {
    try {
        fs.accessSync(filePath, fs.constants.X_OK);
        return true;
    } catch {
        try {
            // Fallback for some systems where accessSync might behave unexpectedly
            execFileSync(filePath, ["--version"], {
                timeout: 1000,
                stdio: "ignore",
            });
            return true;
        } catch {
            return false;
        }
    }
}

/**
 * Returns shell-specific configuration prefixes to normalize behavior.
 * This is useful for disabling globbing features that might interfere with commands.
 */
export function getShellConfigPrefix(shellPath?: string): string | null {
    if (process.env.CLAUDE_CODE_SHELL_PREFIX) {
        return process.env.CLAUDE_CODE_SHELL_PREFIX;
    }

    if (shellPath?.includes("bash")) {
        return "shopt -u extglob 2>/dev/null || true";
    } else if (shellPath?.includes("zsh")) {
        return "setopt NO_EXTENDED_GLOB 2>/dev/null || true";
    }
    return null;
}

/**
 * Detects the best available Posix-compatible shell (bash or zsh) on the system.
 */
export async function detectShellPath(): Promise<string> {
    const shellOverride = process.env.CLAUDE_CODE_SHELL;
    if (shellOverride && isExecutable(shellOverride)) {
        return shellOverride;
    }

    const systemShell = process.env.SHELL;
    const searchDirs = ["/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"];
    const candidates: string[] = [];

    if (systemShell && isExecutable(systemShell)) {
        candidates.push(systemShell);
    }

    // Use 'which' to find shells in the user's PATH
    for (const shell of ['zsh', 'bash']) {
        try {
            const { stdout, code } = await runCommand('which', [shell]);
            if (code === 0 && stdout && isExecutable(stdout)) {
                candidates.push(stdout);
            }
        } catch {
            // Ignore failures in 'which'
        }
    }

    // Search common locations explicitly
    for (const shell of ['zsh', 'bash']) {
        for (const dir of searchDirs) {
            candidates.push(join(dir, shell));
        }
    }

    const found = candidates.find(c => isExecutable(c));
    if (!found) {
        throw new Error("No suitable Posix shell (bash/zsh) found on the system.");
    }
    return found;
}

/**
 * Creates a snapshot of common shell configuration files into a temporary script.
 * This helps the agent run commands in a more predictable environment.
 */
export async function createShellConfigSnapshot(shellPath: string): Promise<string> {
    const tmpDir = os.tmpdir();
    const randomId = Math.random().toString(36).substring(2, 10);
    const fileName = `claude_shell_config_${Date.now()}_${randomId}.sh`;
    const snapshotPath = join(tmpDir, fileName);

    const prefix = getShellConfigPrefix(shellPath);
    let shellConfig = "";

    try {
        // Attempt to aggregate common config files using an interactive subshell
        const cmd = `cat ~/.bashrc ~/.zshrc ~/.profile ~/.bash_profile ~/.zsh_profile 2>/dev/null || true`;
        const { stdout } = await runCommand(shellPath, ["-i", "-c", cmd], { timeout: 3000 });
        shellConfig = stdout;
    } catch (err: any) {
        console.warn(`[Shell] Failed to read shell config for snapshot: ${err.message}`);
    }

    const content = [
        prefix,
        shellConfig,
        "echo 'Claude CLI shell snapshot loaded.'",
    ].filter(Boolean).join("\n");

    fs.writeFileSync(snapshotPath, content, { encoding: "utf8", mode: 0o600 });
    return snapshotPath;
}
export function getGitBashPath(): string | null {
    if (process.platform !== "win32") return null;
    const commonPaths = [
        "C:\\Program Files\\Git\\bin\\bash.exe",
        "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
        join(os.homedir(), "AppData\\Local\\Programs\\Git\\bin\\bash.exe")
    ];
    return commonPaths.find(p => fs.existsSync(p)) || null;
}
